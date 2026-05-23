import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trains.subNets import BertTextEncoder
from trains.subNets.transformer import TransformerEncoder


class DistributionEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class DecAlign(nn.Module):
    """Paper-faithful DecAlign implementation used as the only public model."""

    allowed_datasets = {"mosi", "mosei", "sims", "iemocap"}

    def __init__(self, args):
        super().__init__()
        if args.dataset_name not in self.allowed_datasets:
            allowed = "/".join(sorted(self.allowed_datasets))
            raise ValueError(f"{self.__class__.__name__} is only configured for {allowed}.")

        self.use_bert = args.use_bert
        if self.use_bert:
            self.text_model = BertTextEncoder(
                use_finetune=args.use_finetune,
                transformers=args.transformers,
                pretrained=args.pretrained,
            )

        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        dst_feature_dim, nheads = args.dst_feature_dim_nheads
        self.d_l = self.d_a = self.d_v = dst_feature_dim
        self.d_model = dst_feature_dim
        self.num_heads = nheads
        self.layers = args.nlevels

        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        self.proj_l = nn.Conv1d(
            self.orig_d_l,
            self.d_model,
            kernel_size=args.conv1d_kernel_size_l,
            padding=0,
            bias=False,
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a,
            self.d_model,
            kernel_size=args.conv1d_kernel_size_a,
            padding=0,
            bias=False,
        )
        self.proj_v = nn.Conv1d(
            self.orig_d_v,
            self.d_model,
            kernel_size=args.conv1d_kernel_size_v,
            padding=0,
            bias=False,
        )

        self.encoder_uni_l = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, bias=False)
        self.encoder_uni_a = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, bias=False)
        self.encoder_uni_v = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, bias=False)
        self.encoder_com = nn.Conv1d(self.d_model, self.d_model, kernel_size=1, bias=False)

        self.num_prototypes = args.num_prototypes
        self.gmm_em_iters = getattr(args, "gmm_em_iters", 5)
        self.gmm_var_floor = getattr(args, "gmm_var_floor", 1e-4)
        self.ot_reg = getattr(args, "lambda_ot", 0.1)
        self.ot_num_iters = getattr(args, "ot_num_iters", 50)
        self.hetero_cost_reduction = getattr(args, "hetero_cost_reduction", "mean")
        self.local_proto_mahalanobis = getattr(args, "local_proto_mahalanobis", False)
        self.use_sequence_mask = getattr(args, "use_sequence_mask", False)
        self.use_cross_modal_attention = getattr(args, "use_cross_modal_attention", True)
        active_modalities = getattr(args, "active_modalities", ["text", "audio", "vision"])
        if isinstance(active_modalities, str):
            active_modalities = active_modalities.strip()
            if active_modalities.startswith("[") and active_modalities.endswith("]"):
                active_modalities = active_modalities[1:-1]
            active_modalities = [
                name.strip().strip("'\"") for name in active_modalities.split(",") if name.strip()
            ]
        alias = {"t": "text", "l": "text", "a": "audio", "v": "vision"}
        self.active_modalities = {
            alias.get(str(name).lower(), str(name).lower()) for name in active_modalities
        }
        unknown_modalities = self.active_modalities.difference({"text", "audio", "vision"})
        if unknown_modalities:
            unknown = ", ".join(sorted(unknown_modalities))
            raise ValueError(f"Unknown active_modalities entries: {unknown}")
        if not self.active_modalities:
            raise ValueError("active_modalities must contain at least one modality.")

        self.trans_l = self._build_transformer(self.attn_dropout)
        self.trans_a = self._build_transformer(self.attn_dropout_a)
        self.trans_v = self._build_transformer(self.attn_dropout_v)

        pde_dim = getattr(args, "pde_dim", self.d_model)
        self.pde_l = DistributionEncoder(self.d_model, pde_dim, dropout=self.output_dropout)
        self.pde_a = DistributionEncoder(self.d_model, pde_dim, dropout=self.output_dropout)
        self.pde_v = DistributionEncoder(self.d_model, pde_dim, dropout=self.output_dropout)
        self.mmd_bandwidth = getattr(args, "mmd_bandwidth", 1.0)

        self.out_dropout = nn.Dropout(self.output_dropout)
        output_dim = int(getattr(args, "num_classes", 1)) if args.train_mode == "classification" else 1
        self.out_layer = nn.Linear(6 * self.d_model, output_dim)

    def _build_transformer(self, attn_dropout):
        return TransformerEncoder(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            layers=self.layers,
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def _project(self, x, proj, orig_dim):
        return x if orig_dim == self.d_model else proj(x)

    def _align_mask(self, mask, features):
        if mask is None:
            return None
        mask = mask[:, :, :features.size(2)]
        if mask.size(2) < features.size(2):
            pad = features.size(2) - mask.size(2)
            mask = F.pad(mask, (0, pad), value=0.0)
        return mask.to(device=features.device, dtype=features.dtype)

    def _pool(self, features, mask=None):
        mask = self._align_mask(mask, features)
        if mask is None:
            return features.mean(dim=2)
        denom = mask.sum(dim=2).clamp_min(1.0)
        return (features * mask).sum(dim=2) / denom

    def compute_decoupling_loss(self, s, c):
        batch_size = s.size(0)
        s_flat = s.reshape(batch_size, -1)
        c_flat = c.reshape(batch_size, -1)
        cosine = F.cosine_similarity(s_flat, c_flat, dim=1)
        return cosine.pow(2).mean()

    def _init_gmm(self, x):
        batch_size, dim = x.shape
        k = self.num_prototypes
        if batch_size >= k:
            indices = torch.linspace(0, batch_size - 1, steps=k, device=x.device).round().long()
            mu = x.index_select(0, indices)
        else:
            repeat = math.ceil(k / batch_size)
            mu = x.repeat(repeat, 1)[:k]

        var = x.var(dim=0, unbiased=False).clamp_min(self.gmm_var_floor)
        var = var.unsqueeze(0).expand(k, dim).clone()
        pi = torch.full((k,), 1.0 / k, device=x.device, dtype=x.dtype)
        return pi, mu, var

    def _gmm_log_prob(self, x, pi, mu, var):
        diff = x.unsqueeze(1) - mu.unsqueeze(0)
        mahalanobis = (diff.pow(2) / var.unsqueeze(0)).sum(dim=2)
        log_det = var.log().sum(dim=1).unsqueeze(0)
        log_norm = x.size(1) * math.log(2.0 * math.pi)
        return pi.clamp_min(1e-8).log().unsqueeze(0) - 0.5 * (mahalanobis + log_det + log_norm)

    def _fit_gmm_em(self, x):
        pi, mu, var = self._init_gmm(x)
        for _ in range(self.gmm_em_iters):
            gamma = F.softmax(self._gmm_log_prob(x, pi, mu, var), dim=1)
            nk = gamma.sum(dim=0).clamp_min(1e-6)
            pi = nk / x.size(0)
            mu = gamma.transpose(0, 1).matmul(x) / nk.unsqueeze(1)
            second_moment = gamma.transpose(0, 1).matmul(x.pow(2)) / nk.unsqueeze(1)
            var = (second_moment - mu.pow(2)).clamp_min(self.gmm_var_floor)
        gamma = F.softmax(self._gmm_log_prob(x, pi, mu, var), dim=1)
        return {"pi": pi, "mu": mu, "var": var, "posterior": gamma}

    def _gaussian_cost(self, mu1, var1, mu2, var2):
        diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)
        reduce_fn = torch.sum if self.hetero_cost_reduction == "sum" else torch.mean
        mean_cost = reduce_fn(diff.pow(2), dim=2)
        cov_cost = (
            var1.unsqueeze(1)
            + var2.unsqueeze(0)
            - 2.0 * torch.sqrt(var1.unsqueeze(1) * var2.unsqueeze(0) + 1e-8)
        )
        cov_cost = reduce_fn(cov_cost, dim=2)
        return mean_cost + cov_cost

    def _sample_to_proto_cost(self, x, mu, var):
        diff = x.unsqueeze(1) - mu.unsqueeze(0)
        cost = diff.pow(2)
        if self.local_proto_mahalanobis:
            cost = cost / var.unsqueeze(0).clamp_min(self.gmm_var_floor)
        if self.hetero_cost_reduction == "sum":
            return cost.sum(dim=2)
        return cost.mean(dim=2)

    def _multi_marginal_sinkhorn(self, cost, nu_l, nu_a, nu_v):
        k = cost.size(0)
        kernel = torch.exp(-cost / max(self.ot_reg, 1e-6)).clamp_min(1e-30)
        u = torch.ones(k, device=cost.device, dtype=cost.dtype)
        v = torch.ones_like(u)
        w = torch.ones_like(u)

        for _ in range(self.ot_num_iters):
            u = nu_l / (torch.sum(kernel * v.view(1, k, 1) * w.view(1, 1, k), dim=(1, 2)) + 1e-8)
            v = nu_a / (torch.sum(kernel * u.view(k, 1, 1) * w.view(1, 1, k), dim=(0, 2)) + 1e-8)
            w = nu_v / (torch.sum(kernel * u.view(k, 1, 1) * v.view(1, k, 1), dim=(0, 1)) + 1e-8)

        plan = u.view(k, 1, 1) * v.view(1, k, 1) * w.view(1, 1, k) * kernel
        entropy = (plan * (plan.clamp_min(1e-8).log() - 1.0)).sum()
        return (plan * cost).sum() + self.ot_reg * entropy

    def compute_hetero_loss(self, s_l, s_a, s_v, masks=None):
        mask_l, mask_a, mask_v = masks or (None, None, None)
        x_l = self._pool(s_l, mask_l)
        x_a = self._pool(s_a, mask_a)
        x_v = self._pool(s_v, mask_v)

        g_l = self._fit_gmm_em(x_l)
        g_a = self._fit_gmm_em(x_a)
        g_v = self._fit_gmm_em(x_v)

        nu_l = g_l["posterior"].mean(dim=0)
        nu_a = g_a["posterior"].mean(dim=0)
        nu_v = g_v["posterior"].mean(dim=0)
        nu_l = nu_l / nu_l.sum().clamp_min(1e-8)
        nu_a = nu_a / nu_a.sum().clamp_min(1e-8)
        nu_v = nu_v / nu_v.sum().clamp_min(1e-8)

        cost_la = self._gaussian_cost(g_l["mu"], g_l["var"], g_a["mu"], g_a["var"])
        cost_lv = self._gaussian_cost(g_l["mu"], g_l["var"], g_v["mu"], g_v["var"])
        cost_av = self._gaussian_cost(g_a["mu"], g_a["var"], g_v["mu"], g_v["var"])
        joint_cost = cost_la.unsqueeze(2) + cost_lv.unsqueeze(1) + cost_av.unsqueeze(0)
        ot_loss = self._multi_marginal_sinkhorn(joint_cost, nu_l, nu_a, nu_v)

        local_losses = [
            (g_l["posterior"] * self._sample_to_proto_cost(x_l, g_a["mu"], g_a["var"])).sum(dim=1).mean(),
            (g_l["posterior"] * self._sample_to_proto_cost(x_l, g_v["mu"], g_v["var"])).sum(dim=1).mean(),
            (g_a["posterior"] * self._sample_to_proto_cost(x_a, g_l["mu"], g_l["var"])).sum(dim=1).mean(),
            (g_a["posterior"] * self._sample_to_proto_cost(x_a, g_v["mu"], g_v["var"])).sum(dim=1).mean(),
            (g_v["posterior"] * self._sample_to_proto_cost(x_v, g_l["mu"], g_l["var"])).sum(dim=1).mean(),
            (g_v["posterior"] * self._sample_to_proto_cost(x_v, g_a["mu"], g_a["var"])).sum(dim=1).mean(),
        ]
        local_proto_loss = torch.stack(local_losses).mean()
        return ot_loss + local_proto_loss

    def _distribution_stats(self, z):
        mu = z.mean(dim=0)
        centered = z - mu
        denom = max(z.size(0) - 1, 1)
        cov = centered.transpose(0, 1).matmul(centered) / denom
        var = centered.pow(2).mean(dim=0).clamp_min(1e-6)
        skew = centered.pow(3).mean(dim=0) / var.pow(1.5)
        return mu, cov, skew

    def _stats_distance(self, stats_x, stats_y):
        mu_x, cov_x, skew_x = stats_x
        mu_y, cov_y, skew_y = stats_y
        return (
            F.mse_loss(mu_x, mu_y)
            + F.mse_loss(cov_x, cov_y)
            + F.mse_loss(skew_x, skew_y)
        )

    def compute_mmd(self, x, y):
        x_norm = (x.pow(2).sum(dim=1, keepdim=True))
        y_norm = (y.pow(2).sum(dim=1, keepdim=True))
        dist_xx = x_norm + x_norm.transpose(0, 1) - 2.0 * x.matmul(x.transpose(0, 1))
        dist_yy = y_norm + y_norm.transpose(0, 1) - 2.0 * y.matmul(y.transpose(0, 1))
        dist_xy = x_norm + y_norm.transpose(0, 1) - 2.0 * x.matmul(y.transpose(0, 1))
        gamma = 1.0 / (2.0 * self.mmd_bandwidth)
        return (
            torch.exp(-gamma * dist_xx.clamp_min(0.0)).mean()
            + torch.exp(-gamma * dist_yy.clamp_min(0.0)).mean()
            - 2.0 * torch.exp(-gamma * dist_xy.clamp_min(0.0)).mean()
        )

    def compute_homo_loss(self, c_l, c_a, c_v, masks=None):
        mask_l, mask_a, mask_v = masks or (None, None, None)
        z_l = self.pde_l(self._pool(c_l, mask_l))
        z_a = self.pde_a(self._pool(c_a, mask_a))
        z_v = self.pde_v(self._pool(c_v, mask_v))

        stats_l = self._distribution_stats(z_l)
        stats_a = self._distribution_stats(z_a)
        stats_v = self._distribution_stats(z_v)
        sem_loss = (
            self._stats_distance(stats_l, stats_a)
            + self._stats_distance(stats_l, stats_v)
            + self._stats_distance(stats_a, stats_v)
        ) / 3.0
        mmd_loss = (
            self.compute_mmd(z_l, z_a)
            + self.compute_mmd(z_l, z_v)
            + self.compute_mmd(z_a, z_v)
        ) / 3.0
        return sem_loss + mmd_loss, (z_l, z_a, z_v)

    def _transform_and_pool(self, transformer, features, mask=None, context_features=None):
        sequence = features.permute(2, 0, 1)
        if context_features:
            context = torch.cat([item.permute(2, 0, 1) for item in context_features], dim=0)
            encoded = transformer(sequence, context, context)
        else:
            encoded = transformer(sequence)
        encoded = encoded.permute(1, 2, 0)
        return self._pool(encoded, mask)

    def _cross_context(self, name, features_by_modality, active):
        if not self.use_cross_modal_attention:
            return None
        context = [
            features
            for modality, features in features_by_modality.items()
            if modality != name and modality in active
        ]
        return context or None

    def _input_masks(self, text_input, audio, video):
        if not self.use_sequence_mask:
            return None, None, None
        text_mask = None
        if text_input.ndim == 3 and text_input.shape[1] == 3 and text_input.dtype in (torch.long, torch.int, torch.int64):
            row_1 = text_input[:, 1, :]
            row_2 = text_input[:, 2, :]
            text_mask = row_1 if row_2.sum() == 0 or row_1.sum() >= row_2.sum() else row_2
            text_mask = text_mask.unsqueeze(1).float()
        audio_mask = audio.abs().sum(dim=2).gt(0).unsqueeze(1).float()
        video_mask = video.abs().sum(dim=2).gt(0).unsqueeze(1).float()
        return text_mask, audio_mask, video_mask

    def forward(self, text, audio, video, is_distill=False):
        input_masks = self._input_masks(text, audio, video)
        if self.use_bert:
            text = self.text_model(text)

        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        proj_l = self._project(x_l, self.proj_l, self.orig_d_l)
        proj_a = self._project(x_a, self.proj_a, self.orig_d_a)
        proj_v = self._project(x_v, self.proj_v, self.orig_d_v)
        mask_l, mask_a, mask_v = [self._align_mask(mask, proj) for mask, proj in zip(input_masks, (proj_l, proj_a, proj_v))]
        if self.use_sequence_mask:
            proj_l = proj_l * mask_l if mask_l is not None else proj_l
            proj_a = proj_a * mask_a if mask_a is not None else proj_a
            proj_v = proj_v * mask_v if mask_v is not None else proj_v

        s_l = self.encoder_uni_l(proj_l)
        s_a = self.encoder_uni_a(proj_a)
        s_v = self.encoder_uni_v(proj_v)
        c_l = self.encoder_com(proj_l)
        c_a = self.encoder_com(proj_a)
        c_v = self.encoder_com(proj_v)

        active = self.active_modalities
        zero = proj_l.new_tensor(0.0)
        dec_terms = []
        if "text" in active:
            dec_terms.append(self.compute_decoupling_loss(s_l, c_l))
        if "audio" in active:
            dec_terms.append(self.compute_decoupling_loss(s_a, c_a))
        if "vision" in active:
            dec_terms.append(self.compute_decoupling_loss(s_v, c_v))
        dec_loss = torch.stack(dec_terms).mean() if dec_terms else zero
        masks = (mask_l, mask_a, mask_v)
        if len(active) == 3:
            hete_loss = self.compute_hetero_loss(s_l, s_a, s_v, masks=masks)
            homo_loss, z_common = self.compute_homo_loss(c_l, c_a, c_v, masks=masks)
        else:
            hete_loss = zero
            homo_loss = zero
            z_common = (None, None, None)

        batch_size = text.size(0)
        empty = proj_l.new_zeros(batch_size, self.d_model)
        s_features = {"text": s_l, "audio": s_a, "vision": s_v}
        h_l = (
            self._transform_and_pool(
                self.trans_l,
                s_l,
                mask_l,
                self._cross_context("text", s_features, active),
            )
            if "text" in active
            else empty
        )
        h_a = (
            self._transform_and_pool(
                self.trans_a,
                s_a,
                mask_a,
                self._cross_context("audio", s_features, active),
            )
            if "audio" in active
            else empty
        )
        h_v = (
            self._transform_and_pool(
                self.trans_v,
                s_v,
                mask_v,
                self._cross_context("vision", s_features, active),
            )
            if "vision" in active
            else empty
        )
        c_l_avg = self._pool(c_l, mask_l) if "text" in active else empty
        c_a_avg = self._pool(c_a, mask_a) if "audio" in active else empty
        c_v_avg = self._pool(c_v, mask_v) if "vision" in active else empty

        final_rep = torch.cat([h_l, h_a, h_v, c_l_avg, c_a_avg, c_v_avg], dim=1)
        output = self.out_layer(self.out_dropout(final_rep))

        return {
            "output_logit": output,
            "dec_loss": dec_loss,
            "hete_loss": hete_loss,
            "homo_loss": homo_loss,
            "s_l": s_l,
            "s_a": s_a,
            "s_v": s_v,
            "c_l": c_l,
            "c_a": c_a,
            "c_v": c_v,
            "z_common": z_common,
            "fusion_rep_hete": torch.cat([h_l, h_a, h_v], dim=1),
            "fusion_rep_homo": torch.cat([c_l_avg, c_a_avg, c_v_avg], dim=1),
            "final_rep": final_rep,
        }
