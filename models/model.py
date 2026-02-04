import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from trains.subNets import BertTextEncoder
from trains.subNets.transformer import TransformerEncoder

class DecAlign(nn.Module):
    def __init__(self, args):
        super(DecAlign, self).__init__()
        # 1. Whether to use BERT encoder
        self.use_bert = args.use_bert
        if self.use_bert:
            self.text_model = BertTextEncoder(use_finetune=args.use_finetune, 
                                                transformers=args.transformers,
                                                pretrained=args.pretrained)
        
        # 2. Set sequence lengths for each modality based on dataset
        if args.dataset_name == 'mosi':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 375
        elif args.dataset_name == 'mosei':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 500, 500
        elif args.dataset_name == 'iemocap':
            if args.need_data_aligned:
                self.len_l, self.len_v, self.len_a = 50, 50, 50
            else:
                self.len_l, self.len_v, self.len_a = 50, 375, 500
        else:
            # Default fallback for unknown datasets
            self.len_l, self.len_v, self.len_a = 50, 500, 500
        
        # 3. Original and target feature dimension settings
        self.orig_d_l, self.orig_d_a, self.orig_d_v = args.feature_dims
        dst_feature_dims, nheads = args.dst_feature_dim_nheads
        self.d_l = self.d_a = self.d_v = dst_feature_dims
        self.num_heads = nheads
        self.layers = args.nlevels

        # Dropout and other hyperparameters
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.embed_dropout = args.embed_dropout
        self.res_dropout = args.res_dropout
        self.output_dropout = args.output_dropout
        self.text_dropout = args.text_dropout
        self.attn_mask = args.attn_mask

        # 4. Temporal convolutional layers: project initial features for each modality
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=args.conv1d_kernel_size_l, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=args.conv1d_kernel_size_a, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=args.conv1d_kernel_size_v, padding=0, bias=False)

        # 5. Modality decoupling: extract modality-specific (unique) and modality-common features
        self.encoder_uni_l = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.encoder_uni_a = nn.Conv1d(self.d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.encoder_uni_v = nn.Conv1d(self.d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.encoder_com   = nn.Conv1d(self.d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        self.proj_cosine_l = nn.Linear(self.d_l * (self.len_l - args.conv1d_kernel_size_l + 1), self.d_l)
        self.proj_cosine_a = nn.Linear(self.d_a * (self.len_a - args.conv1d_kernel_size_a + 1), self.d_a)
        self.proj_cosine_v = nn.Linear(self.d_v * (self.len_v - args.conv1d_kernel_size_v + 1), self.d_v)

        # 6. Heterogeneity alignment: based on GMM prototypes and multi-marginal optimal transport
        # Number of prototypes K
        self.num_prototypes = args.num_prototypes
        # Prototype means and log-variances for each modality (assuming diagonal covariance)
        self.proto_l = nn.Parameter(torch.randn(self.num_prototypes, self.d_l))
        self.proto_a = nn.Parameter(torch.randn(self.num_prototypes, self.d_a))
        self.proto_v = nn.Parameter(torch.randn(self.num_prototypes, self.d_v))
        self.logvar_l = nn.Parameter(torch.zeros(self.num_prototypes, self.d_l))
        self.logvar_a = nn.Parameter(torch.zeros(self.num_prototypes, self.d_a))
        self.logvar_v = nn.Parameter(torch.zeros(self.num_prototypes, self.d_v))
        
        # Multi-marginal optimal transport hyperparameters
        self.ot_reg = args.lambda_ot if hasattr(args, 'lambda_ot') else 0.1  # Regularization coefficient
        self.ot_num_iters = args.ot_num_iters if hasattr(args, 'ot_num_iters') else 50  # Sinkhorn iterations

        # 7. Multimodal fusion - heterogeneity branch: two parallel paths
        self.transformer_fusion = TransformerEncoder(embed_dim=3 * self.d_l,
                                                     num_heads=self.num_heads,
                                                     layers=self.layers,
                                                     attn_dropout=self.attn_dropout,
                                                     relu_dropout=self.relu_dropout,
                                                     res_dropout=self.res_dropout,
                                                     embed_dropout=self.embed_dropout,
                                                     attn_mask=self.attn_mask)
        self.trans_l_with_a = self.get_network(self_type='la')
        self.trans_l_with_v = self.get_network(self_type='lv')
        self.trans_a_with_l = self.get_network(self_type='al')
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_l = self.get_network(self_type='vl')
        self.trans_v_with_a = self.get_network(self_type='va')
        self.trans_l_mem   = self.get_network(self_type='l_mem', layers=3)
        self.trans_a_mem   = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem   = self.get_network(self_type='v_mem', layers=3)

        # Project cross-modal attention path output from [N, 6*d_l] to [N, 3*d_l]
        self.cma_proj = nn.Linear(6 * self.d_l, 3 * self.d_l)
        
        # Dynamic output dimension
        output_dim = 6 if args.dataset_name == 'iemocap' else 1
        self.out_layer = nn.Linear(6 * self.d_l, output_dim)

        # 8. Loss weights
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)


    # -------------------------- Helper Methods --------------------------
    def compute_decoupling_loss(self, s, c):
        # Compute cosine similarity between specific and common features
        N = s.size(0)  # [N, d, T]
        s_flat = s.view(N, -1)
        c_flat = c.view(N, -1)
        cos_sim = F.cosine_similarity(s_flat, c_flat, dim=1)
        return cos_sim.mean()

    def compute_prototypes(self, features, proto, logvar):
        """
        Compute soft assignment weights based on distance between sample features and prototypes (means)
        """
        N, d, T = features.size()
        feat_avg = features.mean(dim=2)  # [N, d]
        diff = feat_avg.unsqueeze(1) - proto.unsqueeze(0)  # [N, K, d]
        dist_sq = (diff ** 2).sum(dim=2)  # [N, K]
        w = F.softmax(-dist_sq, dim=1)
        return w

    def pairwise_cost(self, mu1, logvar1, mu2, logvar2, eps=1e-9):
        """
        Compute pairwise cost between two sets of prototypes: Euclidean distance + covariance matching cost
        """
        # Euclidean distance between means
        diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)  # [K, K, d]
        dist_sq = torch.sum(diff ** 2, dim=2)  # [K, K]
        # Covariance part: assuming diagonal covariance, sigma = exp(logvar)
        sigma1 = torch.exp(logvar1)
        sigma2 = torch.exp(logvar2)
        # Diagonal covariance matching cost
        cov_term = torch.sum(sigma1.unsqueeze(1) + sigma2.unsqueeze(0) - 
                             2 * torch.sqrt(sigma1.unsqueeze(1) * sigma2.unsqueeze(0) + eps), dim=2)
        return dist_sq + cov_term

    def multi_marginal_sinkhorn(self, C, nu_l, nu_a, nu_v, reg, num_iters=50, eps=1e-9):
        """
        Solve the three-modality optimal transport problem using multi-marginal Sinkhorn algorithm
        Args:
          - C: Joint cost tensor of shape [K, K, K]
          - nu_l, nu_a, nu_v: Marginal distributions over prototypes for text, audio, video (vectors, shape [K])
          - reg: Regularization parameter (entropy regularization coefficient)
          - num_iters: Number of Sinkhorn iterations
        Returns:
          - T: Joint transport matrix of shape [K, K, K]
          - ot_loss: Optimal transport loss value
        """
        K = C.size(0)
        # Compute kernel matrix
        K_tensor = torch.exp(-C / reg)  # [K, K, K]
        # Initialize scaling factors
        u = torch.ones(K, device=C.device)
        v = torch.ones(K, device=C.device)
        w = torch.ones(K, device=C.device)
        for _ in range(num_iters):
            u = nu_l / (torch.sum(K_tensor * v.view(1, K, 1) * w.view(1, 1, K), dim=(1,2)) + eps)
            v = nu_a / (torch.sum(K_tensor * u.view(K,1,1) * w.view(1,1,K), dim=(0,2)) + eps)
            w = nu_v / (torch.sum(K_tensor * u.view(K,1,1) * v.view(1,K,1), dim=(0,1)) + eps)
        # Compute joint transport matrix T
        T = (u.view(K,1,1) * v.view(1,K,1) * w.view(1,1,K)) * K_tensor
        # Compute OT loss (including entropy regularization term)
        ot_loss = torch.sum(T * C)
        entropy = - torch.sum(T * torch.log(T + eps))
        ot_loss = ot_loss + 0.001 * reg * entropy
        return T, ot_loss

    def compute_hetero_loss(self, s_l, s_a, s_v):
        """
        Heterogeneity alignment loss:
          - Compute soft assignment weights between samples and prototypes via GMM
          - Construct marginal distributions over prototypes using mean assignments
          - Build joint cost tensor between cross-modal prototypes and solve OT loss via multi-marginal Sinkhorn
          - Also compute local alignment loss between samples and other modality prototypes
        """
        # 1. Compute soft assignment weights for each modality
        w_l = self.compute_prototypes(s_l, self.proto_l, self.logvar_l)  # [N, K]
        w_a = self.compute_prototypes(s_a, self.proto_a, self.logvar_a)  # [N, K]
        w_v = self.compute_prototypes(s_v, self.proto_v, self.logvar_v)  # [N, K]

        # 2. Compute marginal distribution over prototypes (average and normalize)
        nu_l = w_l.mean(dim=0)  # [K]
        nu_a = w_a.mean(dim=0)  # [K]
        nu_v = w_v.mean(dim=0)  # [K]
        eps = 1e-9
        nu_l = nu_l / (nu_l.sum() + eps)
        nu_a = nu_a / (nu_a.sum() + eps)
        nu_v = nu_v / (nu_v.sum() + eps)

        # 3. Construct pairwise cost matrices between modalities
        cost_la = self.pairwise_cost(self.proto_l, self.logvar_l, self.proto_a, self.logvar_a, eps=eps)  # [K, K]
        cost_lv = self.pairwise_cost(self.proto_l, self.logvar_l, self.proto_v, self.logvar_v, eps=eps)  # [K, K]
        cost_av = self.pairwise_cost(self.proto_a, self.logvar_a, self.proto_v, self.logvar_v, eps=eps)  # [K, K]
        # Construct joint cost tensor: sum of pairwise costs for three modalities
        # C[i,j,k] = cost_la[i,j] + cost_lv[i,k] + cost_av[j,k]
        C = cost_la.unsqueeze(2) + cost_lv.unsqueeze(1) + cost_av.unsqueeze(0)  # [K, K, K]

        # 4. Solve optimal transport problem using multi-marginal Sinkhorn
        _, ot_loss = self.multi_marginal_sinkhorn(C, nu_l, nu_a, nu_v, reg=self.ot_reg, num_iters=self.ot_num_iters)

        # 5. Local prototype alignment loss: weighted Euclidean distance between samples and other modality prototypes
        feat_l = s_l.mean(dim=2)  # [N, d]
        feat_a = s_a.mean(dim=2)
        feat_v = s_v.mean(dim=2)
        loss_la = torch.mean(w_l * torch.sum((feat_l.unsqueeze(1) - self.proto_a.unsqueeze(0)) ** 2, dim=2))
        loss_lv = torch.mean(w_l * torch.sum((feat_l.unsqueeze(1) - self.proto_v.unsqueeze(0)) ** 2, dim=2))
        loss_al = torch.mean(w_a * torch.sum((feat_a.unsqueeze(1) - self.proto_l.unsqueeze(0)) ** 2, dim=2))
        loss_av = torch.mean(w_a * torch.sum((feat_a.unsqueeze(1) - self.proto_v.unsqueeze(0)) ** 2, dim=2))
        loss_vl = torch.mean(w_v * torch.sum((feat_v.unsqueeze(1) - self.proto_l.unsqueeze(0)) ** 2, dim=2))
        loss_va = torch.mean(w_v * torch.sum((feat_v.unsqueeze(1) - self.proto_a.unsqueeze(0)) ** 2, dim=2))
        local_proto_loss = loss_la + loss_lv + loss_al + loss_av + loss_vl + loss_va

        hetero_loss = ot_loss + local_proto_loss
        return hetero_loss

    def compute_mmd(self, x, y, kernel_bandwidth=1.0):
        xx = torch.mm(x, x.t())
        yy = torch.mm(y, y.t())
        xy = torch.mm(x, y.t())
        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)
        K_xx = torch.exp(- (rx.t() + rx - 2 * xx) / (2 * kernel_bandwidth))
        K_yy = torch.exp(- (ry.t() + ry - 2 * yy) / (2 * kernel_bandwidth))
        K_xy = torch.exp(- (rx.t() + ry - 2 * xy) / (2 * kernel_bandwidth))
        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd

    def compute_homo_loss(self, c_l, c_a, c_v):
        def compute_stats(c):
            mu = c.mean(dim=(0,2))
            sigma = c.var(dim=(0,2))
            centered = c - mu.view(1, -1, 1)
            skew = (centered ** 3).mean(dim=(0,2)) / (sigma + 1e-6).pow(1.5)
            return mu, sigma, skew

        mu_l, sigma_l, skew_l = compute_stats(c_l)
        mu_a, sigma_a, skew_a = compute_stats(c_a)
        mu_v, sigma_v, skew_v = compute_stats(c_v)
        L_sem = ((mu_l - mu_a).pow(2).sum() + (mu_l - mu_v).pow(2).sum() + (mu_a - mu_v).pow(2).sum() +
                 (sigma_l - sigma_a).pow(2).sum() + (sigma_l - sigma_v).pow(2).sum() + (sigma_a - sigma_v).pow(2).sum() +
                 (skew_l - skew_a).pow(2).sum() + (skew_l - skew_v).pow(2).sum() + (skew_a - skew_v).pow(2).sum())
        c_l_pool = c_l.mean(dim=2)
        c_a_pool = c_a.mean(dim=2)
        c_v_pool = c_v.mean(dim=2)
        mmd_la = self.compute_mmd(c_l_pool, c_a_pool)
        mmd_lv = self.compute_mmd(c_l_pool, c_v_pool)
        mmd_av = self.compute_mmd(c_a_pool, c_v_pool)
        L_mmd = mmd_la + mmd_lv + mmd_av
        homo_loss = L_sem + L_mmd
        return homo_loss

    # -------------------------- Forward Pass --------------------------
    def forward(self, text, audio, video, is_distill=False):
        # 1. Text modality encoding (if using BERT)
        if self.use_bert:
            text = self.text_model(text)
        x_l = F.dropout(text.transpose(1, 2), p=self.text_dropout, training=self.training)
        x_a = audio.transpose(1, 2)
        x_v = video.transpose(1, 2)

        # 2. Initial feature projection
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        # 3. Decoupling: extract modality-specific (s_*) and common (c_*) features
        s_l = self.encoder_uni_l(proj_x_l)  # [N, d_l, T_l']
        s_a = self.encoder_uni_a(proj_x_a)  # [N, d_a, T_a']
        s_v = self.encoder_uni_v(proj_x_v)  # [N, d_v, T_v']
        c_l = self.encoder_com(proj_x_l)
        c_a = self.encoder_com(proj_x_a)
        c_v = self.encoder_com(proj_x_v)

        # 4. Decoupling loss
        dec_loss = (self.compute_decoupling_loss(s_l, c_l) +
                    self.compute_decoupling_loss(s_a, c_a) +
                    self.compute_decoupling_loss(s_v, c_v))

        # 5. Heterogeneity alignment loss (based on s_*, includes multi-marginal OT)
        hete_loss = self.compute_hetero_loss(s_l, s_a, s_v)

        # 6. Homogeneity alignment loss (based on c_*)
        homo_loss = self.compute_homo_loss(c_l, c_a, c_v)

        # 7. Heterogeneity branch - two parallel paths
        ## 7.1 Transformer fusion path: temporal modeling on s_*
        s_l_perm = s_l.permute(2, 0, 1)  # [T_l, N, d_l]
        s_a_perm = s_a.permute(2, 0, 1)  # [T_a, N, d_a]
        s_v_perm = s_v.permute(2, 0, 1)  # [T_v, N, d_v]
        T_target = min(s_l_perm.size(0), s_a_perm.size(0), s_v_perm.size(0))
        s_l_perm = s_l_perm[:T_target, :, :]
        s_a_perm = s_a_perm[:T_target, :, :]
        s_v_perm = s_v_perm[:T_target, :, :]
        fused_hetero_trans = torch.cat([s_l_perm, s_a_perm, s_v_perm], dim=2)  # [T_target, N, 3*d_l]
        trans_out = self.transformer_fusion(fused_hetero_trans)  # [T_target, N, 3*d_l]
        fusion_rep_trans = trans_out[-1]

        ## 7.2 Cross-modal attention path: information interaction via cross-attention modules
        s_l_perm = s_l.permute(2, 0, 1)
        s_a_perm = s_a.permute(2, 0, 1)
        s_v_perm = s_v.permute(2, 0, 1)
        T_target = min(s_l_perm.size(0), s_a_perm.size(0), s_v_perm.size(0))
        s_l_perm = s_l_perm[:T_target, :, :]
        s_a_perm = s_a_perm[:T_target, :, :]
        s_v_perm = s_v_perm[:T_target, :, :]
        # (V,A) -> L
        h_l_with_as = self.trans_l_with_a(s_l_perm, s_a_perm, s_a_perm)
        h_l_with_vs = self.trans_l_with_v(s_l_perm, s_v_perm, s_v_perm)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]
        # (L,V) -> A
        h_a_with_ls = self.trans_a_with_l(s_a_perm, s_l_perm, s_l_perm)
        h_a_with_vs = self.trans_a_with_v(s_a_perm, s_v_perm, s_v_perm)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        last_h_a = h_as[-1]
        # (L,A) -> V
        h_v_with_ls = self.trans_v_with_l(s_v_perm, s_l_perm, s_l_perm)
        h_v_with_as = self.trans_v_with_a(s_v_perm, s_a_perm, s_a_perm)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]
        fusion_rep_cma = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)  # [N, 6*d_l]
        fusion_rep_cma = self.cma_proj(fusion_rep_cma)  # [N, 3*d_l]

        ## 7.3 Homogeneity branch: average pooling over time dimension
        c_l_avg = c_l.mean(dim=2)  # [N, d_l]
        c_a_avg = c_a.mean(dim=2)  # [N, d_a]
        c_v_avg = c_v.mean(dim=2)  # [N, d_v]
        fusion_rep_homo = torch.cat([c_l_avg, c_a_avg, c_v_avg], dim=1)  # [N, 3*d_l]

        ## 7.4 Fuse heterogeneous paths and concatenate with homogeneous
        fusion_rep_hete = fusion_rep_trans + fusion_rep_cma  # [N, 3*d_l]
        final_rep = torch.cat([fusion_rep_hete, fusion_rep_homo], dim=1)  # [N, 6*d_l]
        output = self.out_layer(final_rep)

        res = {
            'output_logit': output,
            'dec_loss': dec_loss,
            'hete_loss': hete_loss,
            'homo_loss': homo_loss,
            's_l': s_l,
            's_a': s_a,
            's_v': s_v,
            'c_l': c_l,
            'c_a': c_a,
            'c_v': c_v,
            'fusion_rep_trans': fusion_rep_trans,
            'fusion_rep_cma': fusion_rep_cma,
            'fusion_rep_hete': fusion_rep_hete,
            'fusion_rep_homo': fusion_rep_homo,
            'final_rep': final_rep
        }
        return res
