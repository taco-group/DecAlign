import os
import random
import numpy as np
import torch

def setup_seed(seed):
    """
    Setup random seed for reproducibility
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def assign_gpu(gpu_ids):
    """
    Assign GPU device
    """
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    
    if len(gpu_ids) == 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("Using CPU")
    else:
        # Set CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print(f"Using GPU: {gpu_ids}")
        
        # Print GPU info
        for i, gpu_id in enumerate(gpu_ids):
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                print(f"GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return device

def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_model(model, path, epoch=None, optimizer=None, scheduler=None):
    """
    Save model checkpoint
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    state = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
    }
    
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(state, path)
    print(f"Model saved to {path}")

def load_model(model, path, optimizer=None, scheduler=None):
    """
    Load model checkpoint
    """
    if not os.path.exists(path):
        print(f"Checkpoint not found at {path}")
        return 0
    
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    print(f"Model loaded from {path}, epoch: {epoch}")
    return epoch

def create_dir(path):
    """
    Create directory if it doesn't exist
    """
    os.makedirs(path, exist_ok=True)
    return path