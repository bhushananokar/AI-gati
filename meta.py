# Stabilized ConvNeXt Meta-Learning - keeping ConvNeXt architecture
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
from collections import OrderedDict, deque
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

# STABILIZED Configuration for ConvNeXt Meta-Learning
stable_convnext_config = {
    'darts_model_path': '/mnt/Test/SC202/trained_models/best_darts_model.pth',
    'data_dir': '/mnt/Test/SC202/IMG_CLASSES',
    'model_dir': '/mnt/Test/SC202/trained_models',
    
    # Model parameters
    'img_size': 224,
    'batch_size': 16,
    
    # STABILIZED: Conservative meta-learning parameters for ConvNeXt
    'meta_lr': 5e-5,              # Very small for large ConvNeXt model
    'inner_lr': 0.003,            # Small inner learning rate
    'meta_epochs': 40,            
    'num_inner_steps': 2,         # Conservative inner steps
    'tasks_per_epoch': 6,         # Fewer tasks for stability
    'k_shot': 3,                  # 3-shot learning
    'query_size': 4,              # Small query sets
    'n_way': 3,                   # 3-way classification
    'first_order': True,
    
    # Stability features
    'gradient_clip': 0.3,         # Conservative gradient clipping
    'warmup_epochs': 8,           # Longer warmup for large model
    'moving_avg_window': 7,       # Longer smoothing window
    'early_stopping_patience': 15,
    'weight_decay': 1e-4,
    'dropout_rate': 0.1,          # Light dropout for regularization
    
    'num_workers': 2,
    'use_cuda': True,
}

# Load data function
def load_data(data_dir, img_size=224, batch_size=32, num_workers=4):
    """Load data for meta-learning"""
    print(f"Loading dataset from {data_dir}")
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val") 
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir]):
        raise ValueError("Invalid dataset structure")
    
    # Conservative transforms for ConvNeXt stability
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.3),  # Light augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    print(f"Dataset loaded with {num_classes} classes: {class_names}")
    return train_loader, val_loader, num_classes, class_names

# Custom LayerNorm for channels-first data
class LayerNormChannelsFirst(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# Stabilized ConvNeXt Block
class StableConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, dropout_rate=0.1):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)  # Add dropout for stability
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=False)
        
        # Smaller layer scale for stability
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
        x = self.dropout(x)  # Apply dropout
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input_x + self.drop_path(x)
        return x

class StableMetaConvNeXt(nn.Module):
    """Stabilized ConvNeXt for meta-learning with reduced complexity"""
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[2, 2, 4, 2], dims=[64, 128, 256, 384],  # Smaller than original
                 drop_path_rate=0.1, layer_scale_init_value=1e-6, dropout_rate=0.1):
        super().__init__()
        
        self.depths = depths
        self.dims = dims
        self.num_classes = num_classes
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        
        # Stem with batch norm for stability
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4, bias=False),
            LayerNormChannelsFirst(dims[0])
        )
        self.downsample_layers.append(stem)
        
        # Downsampling layers
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNormChannelsFirst(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2, bias=False),
            )
            self.downsample_layers.append(downsample_layer)
        
        # Feature extraction stages with progressive dropout
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                block = StableConvNeXtBlock(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value,
                    dropout_rate=dropout_rate
                )
                stage_blocks.append(block)
            self.stages.append(nn.ModuleList(stage_blocks))
            cur += depths[i]
        
        # Classifier with dropout
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(dims[-1], num_classes, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNormChannelsFirst)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for block in self.stages[i]:
                x = block(x)
        
        # Global average pooling
        x = x.mean([-2, -1])
        x = self.norm(x)
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.dropout(x)
        x = self.head(x)
        return x
    
    def clone(self):
        """Create a deep copy for inner loop"""
        clone = StableMetaConvNeXt(
            in_chans=3,
            num_classes=self.num_classes,
            depths=self.depths,
            dims=self.dims,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=self.layer_scale_init_value
        )
        clone.load_state_dict(self.state_dict())
        return clone

# Task creation (same as before)
def create_stable_tasks(data_loader, num_tasks, n_way=3, k_shot=3, query_size=4, seed=None):
    """Create stable few-shot tasks"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    print("Collecting data by class...")
    class_data = {}
    
    for inputs, labels in tqdm(data_loader, desc="Loading data", leave=False):
        for i, label in enumerate(labels):
            label_item = label.item()
            if label_item not in class_data:
                class_data[label_item] = []
            
            if len(class_data[label_item]) < (k_shot + query_size + 5):
                class_data[label_item].append(inputs[i].clone())
    
    # Filter classes with sufficient data
    min_samples = k_shot + query_size
    valid_classes = [c for c, data in class_data.items() if len(data) >= min_samples]
    
    print(f"Found {len(valid_classes)} classes with ≥{min_samples} samples")
    
    if len(valid_classes) < n_way:
        raise ValueError(f"Need ≥{n_way} classes, only found {len(valid_classes)}")
    
    tasks = []
    for task_idx in range(num_tasks):
        task_classes = np.random.choice(valid_classes, size=n_way, replace=False)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for new_label, original_class in enumerate(task_classes):
            available_data = class_data[original_class]
            indices = np.random.choice(
                len(available_data), 
                size=min(k_shot + query_size, len(available_data)), 
                replace=False
            )
            
            # Support set
            for i in range(k_shot):
                if i < len(indices):
                    support_data.append(available_data[indices[i]])
                    support_labels.append(new_label)
            
            # Query set
            for i in range(k_shot, k_shot + query_size):
                if i < len(indices):
                    query_data.append(available_data[indices[i]])
                    query_labels.append(new_label)
        
        if len(support_data) > 0 and len(query_data) > 0:
            support_data = torch.stack(support_data)
            support_labels = torch.tensor(support_labels)
            query_data = torch.stack(query_data)
            query_labels = torch.tensor(query_labels)
            
            tasks.append((support_data, support_labels, query_data, query_labels))
    
    print(f"Created {len(tasks)} tasks")
    return tasks

# Stabilized MAML for ConvNeXt
class StabilizedConvNeXtMAML:
    def __init__(self, model, inner_lr=0.003, meta_lr=5e-5, num_inner_steps=2, 
                 gradient_clip=0.3, warmup_epochs=8, weight_decay=1e-4):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.base_meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        
        # Conservative optimizer for large ConvNeXt model
        self.meta_optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=meta_lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Gentle learning rate schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=60, eta_min=meta_lr/20
        )
        
        # Extended history tracking for smoothing
        self.loss_history = deque(maxlen=100)
        self.acc_history = deque(maxlen=100)
        
    def get_lr_scale(self, epoch):
        """Conservative warmup for ConvNeXt"""
        if epoch < self.warmup_epochs:
            return 0.1 + 0.9 * (epoch / self.warmup_epochs)  # Start at 10% LR
        return 1.0
    
    def inner_loop(self, support_data, support_labels, criterion, device):
        """Conservative inner loop for ConvNeXt"""
        fast_model = self.model.clone().to(device)
        fast_model.train()
        
        inner_losses = []
        
        # Use smaller inner learning rate for ConvNeXt stability
        effective_inner_lr = self.inner_lr
        
        for step in range(self.num_inner_steps):
            outputs = fast_model(support_data)
            loss = criterion(outputs, support_labels)
            inner_losses.append(loss.item())
            
            # Compute gradients
            gradients = torch.autograd.grad(
                loss, fast_model.parameters(),
                create_graph=True, retain_graph=False,
                allow_unused=True
            )
            
            # Conservative gradient updates with clipping
            with torch.no_grad():
                for param, grad in zip(fast_model.parameters(), gradients):
                    if grad is not None:
                        # Individual gradient clipping
                        grad = torch.clamp(grad, -self.gradient_clip, self.gradient_clip)
                        param.subtract_(effective_inner_lr * grad)
                        
                        # Adaptive learning rate reduction for stability
                        if step > 0 and inner_losses[-1] > inner_losses[-2]:
                            effective_inner_lr *= 0.8
        
        return fast_model, inner_losses
    
    def meta_step(self, batch_tasks, criterion, device, epoch=0):
        """Stabilized meta step for ConvNeXt"""
        self.model.train()
        meta_losses = []
        task_accuracies = []
        inner_losses_all = []
        
        # Conservative learning rate scaling
        lr_scale = self.get_lr_scale(epoch)
        
        for task_idx, (support_data, support_labels, query_data, query_labels) in enumerate(batch_tasks):
            try:
                support_data = support_data.to(device)
                support_labels = support_labels.to(device)
                query_data = query_data.to(device)
                query_labels = query_labels.to(device)
                
                # Inner loop adaptation
                fast_model, inner_losses = self.inner_loop(
                    support_data, support_labels, criterion, device
                )
                inner_losses_all.extend(inner_losses)
                
                # Query evaluation
                fast_model.eval()
                with torch.set_grad_enabled(True):
                    query_outputs = fast_model(query_data)
                    query_loss = criterion(query_outputs, query_labels)
                
                meta_losses.append(query_loss)
                
                # Calculate accuracy
                with torch.no_grad():
                    _, preds = torch.max(query_outputs, 1)
                    accuracy = (preds == query_labels).float().mean()
                    task_accuracies.append(accuracy.item())
                
            except Exception as e:
                print(f"Error in task {task_idx}: {e}")
                continue
        
        if len(meta_losses) > 0:
            meta_loss = torch.stack(meta_losses).mean()
            
            # Meta optimization with careful learning rate control
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            
            # Conservative gradient clipping for ConvNeXt
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Apply learning rate scaling
            for param_group in self.meta_optimizer.param_groups:
                param_group['lr'] = self.base_meta_lr * lr_scale
            
            self.meta_optimizer.step()
            
            # Track metrics
            avg_accuracy = np.mean(task_accuracies) if task_accuracies else 0
            avg_inner_loss = np.mean(inner_losses_all) if inner_losses_all else 0
            
            self.loss_history.append(meta_loss.item())
            self.acc_history.append(avg_accuracy)
            
            return meta_loss.item(), avg_accuracy, avg_inner_loss
        
        return 0.0, 0.0, 0.0
    
    def get_smoothed_metrics(self, window=7):
        """Get smoothed metrics with configurable window"""
        if len(self.loss_history) > 0 and len(self.acc_history) > 0:
            smooth_loss = np.mean(list(self.loss_history)[-window:])
            smooth_acc = np.mean(list(self.acc_history)[-window:])
            return smooth_loss, smooth_acc
        return 0.0, 0.0

# Main training function for ConvNeXt meta-learning
def run_stable_convnext_meta_learning(config):
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, num_classes, class_names = load_data(
        data_dir=config['data_dir'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create stabilized ConvNeXt model
    model = StableMetaConvNeXt(
        in_chans=3,
        num_classes=config['n_way'],
        depths=[2, 2, 4, 2],  # Smaller than original ConvNeXt
        dims=[64, 128, 256, 384],  # Reduced dims for stability
        drop_path_rate=config.get('dropout_rate', 0.1),
        layer_scale_init_value=1e-6,
        dropout_rate=config.get('dropout_rate', 0.1)
    ).to(device)
    
    print(f"ConvNeXt Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize stabilized MAML for ConvNeXt
    maml = StabilizedConvNeXtMAML(
        model=model,
        inner_lr=config['inner_lr'],
        meta_lr=config['meta_lr'],
        num_inner_steps=config['num_inner_steps'],
        gradient_clip=config['gradient_clip'],
        warmup_epochs=config['warmup_epochs'],
        weight_decay=config['weight_decay']
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for stability
    best_accuracy = 0.0
    patience_counter = 0
    
    # Training history
    history = {
        'meta_loss': [],
        'task_acc': [],
        'smoothed_loss': [],
        'smoothed_acc': []
    }
    
    print(f"Starting ConvNeXt meta-learning for {config['meta_epochs']} epochs")
    print(f"Config: {config['n_way']}-way, {config['k_shot']}-shot, {config['query_size']} queries")
    
    # Training loop
    for epoch in range(config['meta_epochs']):
        epoch_start = time.time()
        print(f"\n=== Epoch {epoch+1}/{config['meta_epochs']} ===")
        
        # Create tasks
        train_tasks = create_stable_tasks(
            train_loader,
            num_tasks=config['tasks_per_epoch'],
            n_way=config['n_way'],
            k_shot=config['k_shot'],
            query_size=config['query_size'],
            seed=epoch
        )
        
        if len(train_tasks) == 0:
            print("No valid tasks created, skipping epoch")
            continue
        
        # Meta training step
        meta_loss, task_acc, inner_loss = maml.meta_step(
            train_tasks, criterion, device, epoch
        )
        
        # Get smoothed metrics
        smooth_loss, smooth_acc = maml.get_smoothed_metrics(
            window=config['moving_avg_window']
        )
        
        # Record history
        history['meta_loss'].append(meta_loss)
        history['task_acc'].append(task_acc)
        history['smoothed_loss'].append(smooth_loss)
        history['smoothed_acc'].append(smooth_acc)
        
        # Step scheduler after warmup
        if epoch >= config['warmup_epochs']:
            maml.scheduler.step()
        
        current_lr = maml.meta_optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Comprehensive logging
        print(f"Epoch {epoch+1} ({epoch_time:.1f}s):")
        print(f"  Raw - Loss: {meta_loss:.4f}, Acc: {task_acc:.4f}")
        print(f"  Smoothed - Loss: {smooth_loss:.4f}, Acc: {smooth_acc:.4f}")
        print(f"  Inner Loss: {inner_loss:.4f}, LR: {current_lr:.6f}")
        
        # Early stopping based on smoothed accuracy
        if smooth_acc > best_accuracy:
            best_accuracy = smooth_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'accuracy': smooth_acc,
                'config': config,
                'history': history
            }, os.path.join(config['model_dir'], 'best_convnext_meta_model.pth'))
            print(f"  ✓ New best ConvNeXt accuracy: {smooth_acc:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"Early stopping after {patience_counter} epochs without improvement")
            break
        
        # Plot progress
        if (epoch + 1) % 10 == 0:
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['meta_loss'], alpha=0.3, label='Raw Loss')
            plt.plot(history['smoothed_loss'], label='Smoothed Loss')
            plt.title('ConvNeXt Meta Loss')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history['task_acc'], alpha=0.3, label='Raw Accuracy')
            plt.plot(history['smoothed_acc'], label='Smoothed Accuracy')
            plt.title('ConvNeXt Task Accuracy')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(config['model_dir'], f'convnext_progress_epoch_{epoch+1}.png'))
            plt.show()
    
    print(f"\nConvNeXt Meta-Learning completed!")
    print(f"Best smoothed accuracy: {best_accuracy:.4f}")
    
    return model, history

# Usage - This keeps ConvNeXt architecture with stability improvements
model, history = run_stable_convnext_meta_learning(stable_convnext_config)
