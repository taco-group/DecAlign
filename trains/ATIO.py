import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
from utils.metrices import MetricsTop

__all__ = ["ATIO"]

logger = logging.getLogger('DecAlign')

class ATIO():
    def __init__(self):
        pass

    def getTrain(self, args):
        return {
            'decalign': DecAlignTrainer,
        }[args.model_name]


class DecAlignTrainer():
    def __init__(self, args):
        self.args = args
        if args.train_mode == 'classification':
            class_weights = getattr(args, 'class_weights', None)
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
            self.criterion = nn.CrossEntropyLoss()
        task_loss = getattr(args, 'task_loss', 'mae').lower()
        if args.train_mode == 'classification':
            pass
        elif task_loss == 'mse':
            self.criterion = nn.MSELoss()
        elif task_loss in ('smooth_l1', 'huber'):
            self.criterion = nn.SmoothL1Loss(beta=getattr(args, 'smooth_l1_beta', 1.0))
        else:
            self.criterion = nn.L1Loss()
        self.metrics = MetricsTop(args.train_mode).getMetrics(args.dataset_name)

    def _metric_mode(self, metric_name):
        metric_name = (metric_name or "").lower()
        if any(token in metric_name for token in ("acc", "f1", "corr")):
            return "max"
        return "min"

    def _metric_value(self, results, metric_name):
        if metric_name in ("F1Acc_score", "F1_Acc_score", "F1Acc"):
            return 0.55 * float(results["F1_score"]) + 0.45 * float(results["Acc_2"])
        if metric_name and metric_name in results:
            return float(results[metric_name])
        dataset_loss_key = self.args.dataset_name.upper()
        return float(results[dataset_loss_key])

    def _is_better(self, current, best, mode):
        if mode == "max":
            return current > best
        return current < best

    def _main_loss(self, logits, labels):
        if self.args.train_mode == 'classification':
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            return F.cross_entropy(logits, labels.view(-1).long(), weight=weight)
        regression_loss = self.criterion(logits.view(-1), labels.view(-1))
        binary_task_weight = getattr(self.args, 'binary_task_weight', 0.0)
        if binary_task_weight <= 0:
            return regression_loss

        flat_logits = logits.view(-1)
        flat_labels = labels.view(-1)
        non_zero = flat_labels != 0
        if not torch.any(non_zero):
            return regression_loss
        binary_targets = (flat_labels[non_zero] > 0).float()
        binary_loss = F.binary_cross_entropy_with_logits(flat_logits[non_zero], binary_targets)
        return regression_loss + binary_task_weight * binary_loss

    def _combined_loss(self, main_loss, outputs):
        dec_loss = outputs.get('dec_loss', 0)
        hete_loss = outputs.get('hete_loss', 0)
        homo_loss = outputs.get('homo_loss', 0)

        return (
            main_loss
            + getattr(self.args, 'dec_loss_weight', 1.0) * dec_loss
            + getattr(self.args, 'alpha', 0.1) * hete_loss
            + getattr(self.args, 'beta', 0.1) * homo_loss
        )

    def _contrastive_loss(self, features, labels):
        if features is None or labels is None:
            return features.new_tensor(0.0) if features is not None else torch.tensor(0.0)

        labels = labels.view(-1)
        if self.args.train_mode == 'classification':
            labels = labels.long()
        else:
            labels = torch.round(torch.clamp(labels, min=-3.0, max=3.0)).long() + 3
        features = torch.nn.functional.normalize(features, dim=1)
        logits = torch.matmul(features, features.transpose(0, 1))
        logits = logits / getattr(self.args, 'ct_temperature', 0.1)

        batch_size = labels.size(0)
        logits_mask = torch.ones_like(logits, dtype=torch.bool)
        logits_mask.fill_diagonal_(False)
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & logits_mask
        valid_rows = positive_mask.sum(dim=1) > 0
        if not torch.any(valid_rows):
            return logits.new_tensor(0.0)

        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * logits_mask.float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
        positive_log_prob = (positive_mask.float() * log_prob).sum(dim=1)
        positive_log_prob = positive_log_prob / positive_mask.sum(dim=1).clamp_min(1)
        return -positive_log_prob[valid_rows].mean()

    def do_train(self, model, dataloader, return_epoch_results=False):
        """Training function"""
        device = next(model.parameters()).device
        
        weight_decay = getattr(self.args, 'weight_decay', 0.0)
        bert_learning_rate = getattr(self.args, 'bert_learning_rate', None)
        if bert_learning_rate is not None and hasattr(model, 'text_model'):
            bert_params = list(model.text_model.parameters())
            bert_param_ids = {id(param) for param in bert_params}
            other_params = [param for param in model.parameters() if id(param) not in bert_param_ids]
            optimizer = optim.Adam([
                {'params': bert_params, 'weight_decay': weight_decay, 'lr': bert_learning_rate},
                {'params': other_params, 'weight_decay': weight_decay, 'lr': self.args.learning_rate},
            ])
        else:
            optimizer = optim.Adam([
                {'params': model.parameters(), 'weight_decay': weight_decay}
            ], lr=self.args.learning_rate)

        early_stop_patience = getattr(self.args, 'early_stop_patience', None)
        if early_stop_patience is None:
            early_stop_patience = getattr(self.args, 'patience', 20)
        scheduler_patience = getattr(self.args, 'scheduler_patience', None)
        if scheduler_patience is None:
            scheduler_patience = getattr(self.args, 'patience', 20)
        selection_metric = getattr(self.args, 'selection_metric', None)
        if selection_metric is None:
            if self.args.train_mode == 'classification':
                selection_metric = 'WAF1' if self.args.dataset_name == 'iemocap' else 'F1_score'
            else:
                selection_metric = self.args.dataset_name.upper()
        scheduler_metric = getattr(self.args, 'scheduler_metric', None)
        if scheduler_metric is None:
            scheduler_metric = selection_metric
        selection_mode = getattr(self.args, 'selection_mode', self._metric_mode(selection_metric))
        scheduler_mode = getattr(self.args, 'scheduler_mode', self._metric_mode(scheduler_metric))
        factor = getattr(self.args, 'factor', 0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            patience=scheduler_patience,
            factor=factor
        )

        best_valid = -1e8 if selection_mode == "max" else 1e8
        best_valid_results = None
        epochs, best_epoch = 0, 0
        model_path = getattr(self.args, 'model_save_path', './pt/decalign.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        while True: 
            epochs += 1
            # Training
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            
            for i_batch, batch_data in enumerate(dataloader['train']):
                text, audio, video = batch_data['text'].to(device), batch_data['audio'].to(device), batch_data['vision'].to(device)
                labels = batch_data['labels']['M'].to(device)

                model.zero_grad()

                outputs = model(text, audio, video)
                logits = outputs['output_logit']
                
                # Main task loss
                main_loss = self._main_loss(logits, labels)
                
                # Combined loss
                loss = self._combined_loss(main_loss, outputs)
                ct_weight = getattr(self.args, 'ct_weight', 0.0)
                if ct_weight > 0:
                    ct_loss = self._contrastive_loss(outputs.get('final_rep'), labels)
                    loss = loss + ct_weight * ct_loss

                loss.backward()

                # Gradient clipping
                if hasattr(self.args, 'clip'):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
                
                optimizer.step()
                
                train_loss += loss.item()
                y_pred.append(logits.cpu())
                y_true.append(labels.cpu())

            train_loss = train_loss / len(dataloader['train'])
            
            logger.info(f"TRAIN-({epochs}) loss: {train_loss:.4f}")
            
            # Validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            val_score = self._metric_value(val_results, selection_metric)
            scheduler_score = self._metric_value(val_results, scheduler_metric)
            
            scheduler.step(scheduler_score)
            
            # Save best model by the configured validation metric.
            if self._is_better(val_score, best_valid, selection_mode):
                best_valid, best_epoch = val_score, epochs
                best_valid_results = dict(val_results)
                best_valid_results['epoch'] = epochs
                best_valid_results['selection_metric'] = selection_metric
                best_valid_results['selection_score'] = val_score
                torch.save(model.state_dict(), model_path)
                    
            # Early stopping
            if epochs - best_epoch >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epochs}")
                break

            if epochs >= getattr(self.args, 'num_epochs', 100):
                logger.info(f"Reached maximum epochs {epochs}")
                break

        if best_valid_results is not None:
            cur_seed = getattr(self.args, 'cur_seed', '')
            logger.info(f"BEST_VAL-({cur_seed}) >> {best_valid_results}")
        logger.info(
            f"Best {self.args.dataset_name} on validation "
            f"({selection_metric}, {selection_mode}): {best_valid} at epoch {best_epoch}"
        )
        return best_valid_results

    def do_test(self, model, dataloader, mode="VAL"):
        """Testing/Validation function"""
        model.eval()
        device = next(model.parameters()).device
        
        y_pred, y_true = [], []
        eval_loss = 0.0
        
        with torch.no_grad():
            for i_batch, batch_data in enumerate(dataloader):
                text, audio, video = batch_data['text'].to(device), batch_data['audio'].to(device), batch_data['vision'].to(device)
                labels = batch_data['labels']['M'].to(device)

                outputs = model(text, audio, video)
                logits = outputs['output_logit']
                
                loss = self._main_loss(logits, labels)
                eval_loss += loss.item()
                
                y_pred.append(logits.cpu())
                y_true.append(labels.cpu())
        
        eval_loss = eval_loss / len(dataloader)
        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        
        eval_results = self.metrics(y_pred, y_true)
        eval_results[self.args.dataset_name.upper()] = eval_loss

        cur_seed = getattr(self.args, 'cur_seed', '')
        logger.info(f"{mode}-({cur_seed}) >> {eval_results}")
        return eval_results
