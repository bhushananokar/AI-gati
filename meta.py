# Fixed meta-learning implementation addressing key issues
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
import time
import random
from tqdm.notebook import tqdm
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# FIXED Configuration with better hyperparameters
meta_config = {
    'darts_model_path': '/mnt/Test/SC202/trained_models/best_darts_model.pth',
    'data_dir': '/mnt/Test/SC202/IMG_CLASSES',
    'model_dir': '/mnt/Test/SC202/trained_models',
    
    # Model parameters
    'img_size': 224,
    'batch_size': 16,
    
    # FIXED: Better meta-learning parameters
    'meta_lr': 1e-3,              # Increased from 5e-6
    'inner_lr': 0.01,             # Increased from 0.0005  
    'meta_epochs': 20,            # More epochs
    'num_inner_steps': 3,         # Increased from 1
    'tasks_per_epoch': 10,        # More tasks per epoch
    'k_shot': 3,                  # Samples per class for support
    'query_size': 5,              # Samples per class for query
    'n_way': 3,                   # Number of classes per task
    'first_order': True,
    
    'num_workers': 4,
    'use_cuda': True,
}

# Simplified model without DARTS complexity for meta-learning
class SimpleConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input_x + self.drop_path(x)
        return x

class MetaLearningConvNeXt(nn.Module):
    """Simplified ConvNeXt for meta-learning without DARTS complexity"""
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[2, 2, 6, 2], dims=[64, 128, 256, 512]):
        super().__init__()
        
        self.depths = depths
        self.dims = dims
        self.num_classes = num_classes
        
        # Stem
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        # Feature extraction stages
        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[SimpleConvNeXtBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        # Classifier
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # Global average pooling
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def clone(self):
        """Create a deep copy for inner loop"""
        clone = MetaLearningConvNeXt(
            in_chans=3,
            num_classes=self.num_classes,
            depths=self.depths,
            dims=self.dims
        )
        clone.load_state_dict(self.state_dict())
        return clone

# FIXED: Proper few-shot task creation
def create_few_shot_tasks(data_loader, num_tasks, n_way=3, k_shot=3, query_size=5):
    """Create proper few-shot learning tasks with correct label mapping"""
    
    # Collect all data by class
    print("Collecting data by class...")
    class_data = {}
    
    for inputs, labels in tqdm(data_loader, desc="Loading data"):
        for i, label in enumerate(labels):
            label_item = label.item()
            if label_item not in class_data:
                class_data[label_item] = []
            if len(class_data[label_item]) < (k_shot + query_size + 10):  # Buffer
                class_data[label_item].append(inputs[i])
    
    # Filter classes with enough samples
    valid_classes = [c for c, data in class_data.items() 
                    if len(data) >= (k_shot + query_size)]
    
    print(f"Found {len(valid_classes)} classes with enough samples")
    
    if len(valid_classes) < n_way:
        raise ValueError(f"Need at least {n_way} classes, only found {len(valid_classes)}")
    
    tasks = []
    for task_idx in range(num_tasks):
        # Sample random classes for this task
        task_classes = np.random.choice(valid_classes, size=n_way, replace=False)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        # FIXED: Proper label mapping (0, 1, 2, ... for each task)
        for new_label, original_class in enumerate(task_classes):
            class_samples = class_data[original_class]
            
            # Randomly sample from available data
            indices = np.random.choice(len(class_samples), 
                                     size=k_shot + query_size, 
                                     replace=False)
            
            # Support set
            for i in range(k_shot):
                support_data.append(class_samples[indices[i]])
                support_labels.append(new_label)  # FIXED: Use remapped labels
            
            # Query set  
            for i in range(k_shot, k_shot + query_size):
                query_data.append(class_samples[indices[i]])
                query_labels.append(new_label)  # FIXED: Use remapped labels
        
        # Convert to tensors
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels)
        
        tasks.append((support_data, support_labels, query_data, query_labels))
    
    return tasks

# FIXED: Improved MAML implementation
class ImprovedMAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=3, first_order=True):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        
        # Better optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=meta_lr,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=50
        )
    
    def inner_loop(self, support_data, support_labels, criterion, device):
        """Improved inner loop with better adaptation"""
        # Clone model for inner loop
        fast_model = self.model.clone().to(device)
        fast_model.train()
        
        # Track inner loop progress
        inner_losses = []
        
        for step in range(self.num_inner_steps):
            outputs = fast_model(support_data)
            loss = criterion(outputs, support_labels)
            inner_losses.append(loss.item())
            
            # Compute gradients
            gradients = torch.autograd.grad(
                loss, 
                fast_model.parameters(),
                create_graph=not self.first_order,
                retain_graph=not self.first_order
            )
            
            # Manual parameter update
            with torch.no_grad():
                for param, grad in zip(fast_model.parameters(), gradients):
                    if grad is not None:
                        param.subtract_(self.inner_lr * grad)
        
        return fast_model, inner_losses
    
    def meta_step(self, batch_tasks, criterion, device):
        """Improved meta step with better error handling"""
        self.model.train()
        meta_loss = 0.0
        task_accuracies = []
        inner_losses_all = []
        
        valid_tasks = 0
        
        for task_idx, (support_data, support_labels, query_data, query_labels) in enumerate(batch_tasks):
            try:
                # Move to device
                support_data = support_data.to(device)
                support_labels = support_labels.to(device) 
                query_data = query_data.to(device)
                query_labels = query_labels.to(device)
                
                # Debug prints
                print(f"Task {task_idx}: Support shape {support_data.shape}, "
                      f"labels {support_labels.unique()}")
                
                # Inner loop adaptation
                fast_model, inner_losses = self.inner_loop(
                    support_data, support_labels, criterion, device
                )
                inner_losses_all.extend(inner_losses)
                
                # Evaluate on query set
                fast_model.eval()
                query_outputs = fast_model(query_data)
                query_loss = criterion(query_outputs, query_labels)
                
                # Calculate accuracy
                with torch.no_grad():
                    _, preds = torch.max(query_outputs, 1)
                    accuracy = (preds == query_labels).float().mean()
                    task_accuracies.append(accuracy.item())
                
                print(f"Task {task_idx}: Inner losses {inner_losses}, "
                      f"Query loss {query_loss.item():.4f}, Acc {accuracy.item():.4f}")
                
                meta_loss += query_loss
                valid_tasks += 1
                
            except Exception as e:
                print(f"Error in task {task_idx}: {e}")
                continue
        
        if valid_tasks > 0:
            meta_loss = meta_loss / valid_tasks
            
            # Meta optimization
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.meta_optimizer.step()
            
            avg_accuracy = np.mean(task_accuracies) if task_accuracies else 0
            avg_inner_loss = np.mean(inner_losses_all) if inner_losses_all else 0
            
            return meta_loss.item(), avg_accuracy, avg_inner_loss
        
        return 0.0, 0.0, 0.0

# FIXED: Main training function with better monitoring
def run_improved_meta_learning(config):
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, num_classes, class_names = load_data(
        data_dir=config['data_dir'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create simplified model
    model = MetaLearningConvNeXt(
        in_chans=3,
        num_classes=config['n_way'],  # Use n_way for few-shot learning
        depths=[2, 2, 4, 2],  # Smaller model
        dims=[64, 128, 256, 512]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize MAML
    maml = ImprovedMAML(
        model=model,
        inner_lr=config['inner_lr'],
        meta_lr=config['meta_lr'],
        num_inner_steps=config['num_inner_steps'],
        first_order=config['first_order']
    )
    
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    
    # Training loop
    for epoch in range(config['meta_epochs']):
        print(f"\n=== Epoch {epoch+1}/{config['meta_epochs']} ===")
        
        # Create tasks
        train_tasks = create_few_shot_tasks(
            train_loader,
            num_tasks=config['tasks_per_epoch'],
            n_way=config['n_way'],
            k_shot=config['k_shot'],
            query_size=config['query_size']
        )
        
        # Meta training step
        meta_loss, task_acc, inner_loss = maml.meta_step(train_tasks, criterion, device)
        
        print(f"Meta Loss: {meta_loss:.4f}")
        print(f"Task Accuracy: {task_acc:.4f}")
        print(f"Inner Loss: {inner_loss:.4f}")
        
        # Step scheduler
        maml.scheduler.step()
        current_lr = maml.meta_optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if task_acc > best_accuracy:
            best_accuracy = task_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'accuracy': task_acc,
                'config': config
            }, os.path.join(config['model_dir'], 'best_meta_model_fixed.pth'))
            print(f"New best accuracy: {task_acc:.4f}")
    
    return model

# Usage
# model = run_improved_meta_learning(meta_config)
