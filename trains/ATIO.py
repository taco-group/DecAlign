import os
import time
import logging
import torch
import torch.nn as nn
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
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetrics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        """Training function"""
        device = next(model.parameters()).device
        
        weight_decay = getattr(self.args, 'weight_decay', 0.0)
        optimizer = optim.Adam([
            {'params': model.parameters(), 'weight_decay': weight_decay}
        ], lr=self.args.learning_rate)

        patience = getattr(self.args, 'patience', 20)
        factor = getattr(self.args, 'factor', 0.1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor,
            verbose=True
        )

        best_valid = 1e8
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.train_mode == 'regression' else 'max'
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
                main_loss = self.criterion(logits.view(-1), labels.view(-1))
                
                # Auxiliary losses
                dec_loss = outputs.get('dec_loss', 0)
                hete_loss = outputs.get('hete_loss', 0) 
                homo_loss = outputs.get('homo_loss', 0)
                
                # Combined loss
                loss = main_loss + self.args.alpha1 * dec_loss + self.args.alpha2 * (hete_loss + homo_loss)

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
            val_loss = val_results[self.args.dataset_name.upper()]
            
            scheduler.step(val_loss)
            
            # Save best model
            if min_or_max == 'min':
                if val_loss < best_valid:
                    best_valid, best_epoch = val_loss, epochs
                    torch.save(model.state_dict(), model_path)
            else:
                if val_loss > best_valid:
                    best_valid, best_epoch = val_loss, epochs
                    torch.save(model.state_dict(), model_path)
                    
            # Early stopping
            if epochs - best_epoch >= getattr(self.args, 'patience', 20):
                logger.info(f"Early stopping at epoch {epochs}")
                break

            if epochs >= getattr(self.args, 'num_epochs', 100):
                logger.info(f"Reached maximum epochs {epochs}")
                break

        logger.info(f"Best {self.args.dataset_name} on validation: {best_valid} at epoch {best_epoch}")
        return None

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
                
                loss = self.criterion(logits.view(-1), labels.view(-1))
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