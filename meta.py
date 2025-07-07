# Complete meta-learning implementation with fixed in-place operation issue
# This script is fully self-contained and doesn't depend on previous variables

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import os
import json
import time
import random
from tqdm.notebook import tqdm
from collections import OrderedDict
import copy
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

# Configuration
meta_config = {
    # Path to DARTS model - update this to your checkpoint location
    'darts_model_path': '/mnt/Test/SC202/trained_models/best_darts_model.pth',
    
    # Data and model directories
    'data_dir': '/mnt/Test/SC202/IMG_CLASSES',  # Update to your dataset location
    'model_dir': '/mnt/Test/SC202/trained_models',
    
    # Model parameters
    'img_size': 224,
    'batch_size': 16,
    
    # Meta-learning parameters
    'meta_lr': 5e-6,                  # Extremely small learning rate
    'inner_lr': 0.0005,               # Small inner learning rate
    'meta_epochs': 10,                # Total meta-learning epochs
    'num_inner_steps': 1,             # Start with 1 inner step for stability
    'tasks_per_epoch': 3,             # Reduced number of tasks
    'k_shot': 4,                      # Samples per class for support set
    'query_size': 8,                  # Samples per class for query set
    'first_order': True,              # Use first-order approximation
    
    # Hardware configuration
    'num_workers': 4,
    'use_cuda': True,                 # Use CUDA if available
}

# DropPath implementation
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

# Operations for neural architecture search
class Operations(nn.Module):
    def __init__(self, C, stride, affine=True):
        super().__init__()
        self.ops = nn.ModuleDict({
            'none': nn.Identity() if stride == 1 else nn.Sequential(
                nn.Conv2d(C, C, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(C, affine=affine)
            ),
            'skip_connect': nn.Identity() if stride == 1 else nn.Sequential(
                nn.Conv2d(C, C, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(C, affine=affine)
            ),
            'sep_conv_3x3': nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=1, groups=C, bias=False),
                nn.BatchNorm2d(C, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C, affine=affine),
            ),
            'sep_conv_5x5': nn.Sequential(
                nn.Conv2d(C, C, kernel_size=5, stride=stride, padding=2, groups=C, bias=False),
                nn.BatchNorm2d(C, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C, affine=affine),
            ),
            'dil_conv_3x3': nn.Sequential(
                nn.Conv2d(C, C, kernel_size=3, stride=stride, padding=2, dilation=2, groups=C, bias=False),
                nn.BatchNorm2d(C, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(C, C, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(C, affine=affine),
            ),
        })
        
    def forward(self, x, weights=None):
        if weights is None:
            # Process without explicit weights
            out = 0
            for op in self.ops.values():
                out = out + op(x)
            return out / len(self.ops)
        else:
            # Weighted sum based on architecture parameters
            out = 0
            for i, (op_name, op) in enumerate(self.ops.items()):
                out = out + weights[i] * op(x)
            return out

class SearchableBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dim = dim
        self.ops = Operations(dim, stride=1)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, weights=None):
        input_x = x
        
        # Apply operations
        x = self.ops(x, weights)
        
        # Permute: [N, C, H, W] -> [N, H, W, C]
        x = x.permute(0, 2, 3, 1)
        
        # Apply transformations
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
        
        # Permute back: [N, H, W, C] -> [N, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # Apply residual connection and drop path
        x = input_x + self.drop_path(x)
        return x

class ConvNeXtWithSearch(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        
        # Save configuration
        self.depths = depths
        self.dims = dims
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        
        # Stem and downsample layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)
        
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        # Stages
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.blocks = [] # List to store all blocks for easier parameter access
        cur = 0
        
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                block = SearchableBlock(
                    dim=dims[i], 
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value
                )
                stage_blocks.append(block)
                self.blocks.append(block)
            self.stages.append(nn.ModuleList(stage_blocks))
            cur += depths[i]
        
        # Architecture parameters for DARTS
        self.arch_params = nn.ParameterList([
            nn.Parameter(torch.ones(5) / 5) for _ in range(sum(depths))
        ])
        
        # Final normalization and classifier
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def get_arch_parameters(self):
        return self.arch_params
    
    def forward_features(self, x, weights_list=None):
        # Use softmax-normalized architecture weights
        if weights_list is None:
            weights_list = [F.softmax(w, dim=0) for w in self.arch_params]
        
        # Process through stages
        block_idx = 0
        
        for i in range(4):
            # Downsample
            x = self.downsample_layers[i](x)
            
            # Process blocks in stage
            for j, block in enumerate(self.stages[i]):
                # Process block with architecture weights
                x = block(x, weights_list[block_idx])
                block_idx += 1
        
        # Apply global average pooling
        x = x.mean([-2, -1])
        
        # Apply final normalization
        x = self.norm(x)
        return x
    
    def forward(self, x, weights_list=None):
        x = self.forward_features(x, weights_list)
        x = self.head(x)
        return x
    
    def clone(self):
        """Create a deep copy of the model with the same architecture"""
        clone = ConvNeXtWithSearch(
            in_chans=self.in_chans,
            num_classes=self.num_classes,
            depths=self.depths,
            dims=self.dims,
            drop_path_rate=self.drop_path_rate,
            layer_scale_init_value=self.layer_scale_init_value
        )
        clone.load_state_dict(self.state_dict())
        return clone

# Advanced augmentation for dataset loading
class AdvancedAugmentation:
    def __init__(self, img_size=224, color_jitter=0.3, auto_augment=True):
        # Training transforms with advanced augmentation
        train_transforms = []
        
        # Start with resize and center crop
        train_transforms.extend([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
        ])
        
        # Add various augmentations
        train_transforms.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter/2
            ),
        ])
        
        # Use AutoAugment policy if selected
        if auto_augment:
            train_transforms.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        
        # Finalize transforms
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3)
        ])
        
        # Validation transforms - only resize, center crop and normalize
        val_transforms = [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        self.train_transform = transforms.Compose(train_transforms)
        self.val_transform = transforms.Compose(val_transforms)

# Function to load data
def load_data(data_dir, img_size=224, batch_size=32, auto_augment=True, num_workers=4):
    """Load data for meta-learning"""
    print(f"Loading dataset from {data_dir}")
    
    # Check path exists
    if not os.path.exists(data_dir):
        print(f"WARNING: Data directory {data_dir} does not exist!")
        
    # Check if train/val/test splits exist
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val") 
    
    if not all(os.path.exists(d) for d in [train_dir, val_dir]):
        print("ERROR: Could not find train/val splits in data directory")
        print("Please make sure your data is organized with train/val subdirectories")
        raise ValueError("Invalid dataset structure")
    
    # Create augmentation transforms
    augmentation = AdvancedAugmentation(img_size=img_size, auto_augment=auto_augment)
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=train_dir, 
        transform=augmentation.train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=val_dir, 
        transform=augmentation.val_transform
    )
    
    print(f"Found {len(train_dataset)} training images and {len(val_dataset)} validation images")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    print(f"Dataset loaded with {num_classes} classes: {class_names}")
    
    return train_loader, val_loader, num_classes, class_names

# Function to create tasks for meta-learning
def create_tasks(data_loader, num_tasks, k_shot=4, query_size=16):
    """Create episodes for meta-learning from a data loader"""
    tasks = []
    
    # Get total number of classes
    num_classes = len(data_loader.dataset.classes)
    all_classes = list(range(num_classes))
    
    # Progress tracking
    task_pbar = tqdm(total=num_tasks, desc="Creating meta-learning tasks")
    
    # First, collect data by class for faster access
    print("Collecting data by class...")
    class_data = {c: [] for c in all_classes}
    data_pbar = tqdm(data_loader, desc="Loading dataset")
    
    for inputs, labels in data_pbar:
        for i, label in enumerate(labels):
            label_item = label.item()
            if len(class_data[label_item]) < (k_shot + query_size + 5):  # Add some buffer
                class_data[label_item].append(inputs[i].clone())  # Clone to avoid issues
    
    print(f"Data collection complete. Available samples per class:")
    for c in all_classes:
        print(f"  Class {c}: {len(class_data[c])} samples")
    
    # Now create tasks with the collected data
    for task_idx in range(num_tasks):
        # Sample random classes for this task (min of 2 classes or all available)
        min_classes = min(5, num_classes)
        if min_classes < 2:
            print("ERROR: Need at least 2 classes for meta-learning")
            break
            
        task_classes = np.random.choice(all_classes, size=min_classes, replace=False)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        # For progress and debugging
        classes_added = 0
        
        # Create support and query sets
        for c in task_classes:
            # Check if we have enough samples
            if len(class_data[c]) < (k_shot + query_size):
                print(f"Warning: Not enough samples for class {c} - need {k_shot + query_size}, have {len(class_data[c])}")
                continue
                
            # Support set
            for i in range(k_shot):
                support_data.append(class_data[c][i].clone())  # Clone to avoid issues
                support_labels.append(torch.tensor(c))
                
            # Query set
            for i in range(k_shot, k_shot + query_size):
                if i < len(class_data[c]):
                    query_data.append(class_data[c][i].clone())  # Clone to avoid issues
                    query_labels.append(torch.tensor(c))
            
            classes_added += 1
        
        # Only add task if it has at least 2 classes
        if classes_added >= 2 and len(support_data) > 0 and len(query_data) > 0:
            # Convert to tensors
            support_data = torch.stack(support_data)
            support_labels = torch.stack(support_labels)
            query_data = torch.stack(query_data)
            query_labels = torch.stack(query_labels)
            
            tasks.append((support_data, support_labels, query_data, query_labels))
            task_pbar.update(1)
        else:
            print(f"Warning: Task {task_idx} did not have enough valid classes and samples")
    
    task_pbar.close()
    print(f"Created {len(tasks)} meta-learning tasks")
    
    # Check if we have any tasks
    if len(tasks) == 0:
        raise ValueError("No valid tasks could be created. Check your dataset.")
        
    return tasks

# Model-Agnostic Meta-Learning with separate inner model
class MAML:
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=1, first_order=True):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order
        
        # Create a separate model for inner loop to avoid in-place issues
        self.fast_model = None
        
        # Use SGD optimizer with lower learning rate for stability
        self.meta_optimizer = torch.optim.SGD(
            self.model.parameters(), 
            lr=meta_lr,
            momentum=0.9,
            weight_decay=0.01
        )
    
    def inner_loop(self, support_data, support_labels, criterion, device):
        """Perform inner loop adaptation using a separate model instance"""
        # Create a new clone of the model for inner loop
        self.fast_model = self.model.clone().to(device)
        self.fast_model.train()
        
        # Store current params for tracking
        fast_weights = OrderedDict(
            (name, param.clone()) for name, param in self.fast_model.named_parameters()
        )
        
        # Print shape info
        print(f"Support data shape: {support_data.shape}, labels shape: {support_labels.shape}")
        
        # Inner loop updates with progress tracking
        inner_pbar = tqdm(range(self.num_inner_steps), desc="Inner adaptation steps")
        for step in inner_pbar:
            # Forward pass 
            outputs = self.fast_model(support_data)
            loss = criterion(outputs, support_labels)
            inner_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Compute gradients
            gradients = torch.autograd.grad(
                loss, 
                self.fast_model.parameters(),
                create_graph=not self.first_order,
                retain_graph=not self.first_order,
                allow_unused=True
            )
            
            # Update fast model parameters manually
            with torch.no_grad():
                for (name, param), grad in zip(self.fast_model.named_parameters(), gradients):
                    if grad is not None:
                        new_param = param - self.inner_lr * grad
                        param.copy_(new_param)
                        fast_weights[name] = new_param
        
        # Return the adapted model
        return self.fast_model
    
    def meta_step(self, batch_tasks, criterion, device):
        """Perform meta-learning step with separate inner loop model"""
        meta_loss = 0.0
        task_accuracies = []
        
        # Original model weights 
        self.model.train()
        
        # Process each task
        task_pbar = tqdm(batch_tasks, desc="Meta-learning tasks")
        for task_idx, (support_data, support_labels, query_data, query_labels) in enumerate(task_pbar):
            # Move data to device
            support_data = support_data.to(device)
            support_labels = support_labels.to(device)
            query_data = query_data.to(device)
            query_labels = query_labels.to(device)
            
            try:
                # Inner loop adaptation - this uses a separate model
                fast_model = self.inner_loop(support_data, support_labels, criterion, device)
                
                # Evaluate on query set with adapted model
                fast_model.eval()
                with torch.set_grad_enabled(True):  # Keep gradients for meta-update
                    query_outputs = fast_model(query_data)
                    query_loss = criterion(query_outputs, query_labels)
                
                # For reporting accuracy
                with torch.no_grad():
                    _, preds = torch.max(query_outputs, 1)
                    accuracy = torch.sum(preds == query_labels).float() / query_labels.size(0)
                    task_accuracies.append(accuracy.item())
                
                task_pbar.set_postfix({
                    "task": f"{task_idx+1}/{len(batch_tasks)}",
                    "loss": f"{query_loss.item():.4f}",
                    "acc": f"{accuracy.item():.4f}"
                })
                
                # Add to meta loss
                meta_loss = meta_loss + query_loss / len(batch_tasks)
                
            except Exception as e:
                print(f"Error in task {task_idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # Meta optimization
        if meta_loss > 0:
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.meta_optimizer.step()
            
            # Report average accuracy across tasks
            avg_accuracy = np.mean(task_accuracies) if task_accuracies else 0
            print(f"Meta-step completed. Avg task accuracy: {avg_accuracy:.4f}")
            
            return meta_loss.item(), avg_accuracy
        else:
            print("Warning: No tasks processed in meta_step")
            return 0.0, 0.0

# Main function to run meta-learning
def run_meta_learning(config):
    # Device configuration
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model directory if needed
    model_dir = config['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    train_loader, val_loader, num_classes, class_names = load_data(
        data_dir=config['data_dir'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        auto_augment=True,
        num_workers=config['num_workers']
    )
    
    # Load DARTS model checkpoint
    try:
        print(f"Loading checkpoint from {config['darts_model_path']}")
        checkpoint = torch.load(config['darts_model_path'], map_location='cpu')
        print("Checkpoint loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Will initialize model from scratch instead")
        checkpoint = {}
    
    # Initialize model
    print("Initializing ConvNeXt model")
    model = ConvNeXtWithSearch(
        in_chans=3,
        num_classes=checkpoint.get('num_classes', num_classes),
        depths=checkpoint.get('depths', [3, 3, 9, 3]),
        dims=checkpoint.get('dims', [96, 192, 384, 768]),
        drop_path_rate=0.0  # Set to 0 for meta-learning
    )
    
    # Load model weights from checkpoint if available
    if 'model_state_dict' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Successfully loaded model weights from checkpoint")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            print("Initializing with random weights instead")
    else:
        print("No model_state_dict found in checkpoint, using random initialization")
    
    # Move model to device
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Initialize MAML
    maml = MAML(
        model=model,
        inner_lr=config['inner_lr'],
        meta_lr=config['meta_lr'],
        num_inner_steps=config['num_inner_steps'],
        first_order=config.get('first_order', True)
    )
    
    # Training metrics
    best_val_acc = 0.0
    history = {
        'meta_loss': [],
        'task_acc': [],
        'val_acc': []
    }
    
    # Meta-training loop
    print(f"Starting meta-training for {config['meta_epochs']} epochs")
    for epoch in range(config['meta_epochs']):
        epoch_start_time = time.time()
        print(f"\nMeta-training epoch {epoch+1}/{config['meta_epochs']}")
        
        # Create meta-learning tasks
        num_tasks = config.get('tasks_per_epoch', 3)
        print(f"Creating {num_tasks} meta-learning tasks...")
        train_tasks = create_tasks(
            train_loader,
            num_tasks=num_tasks,
            k_shot=config.get('k_shot', 4),
            query_size=config.get('query_size', 8)
        )
        
        if len(train_tasks) == 0:
            print("No valid tasks created. Skipping this epoch.")
            continue
        
        # Meta-training
        model.train()
        meta_loss, task_acc = maml.meta_step(train_tasks, criterion, device)
        history['meta_loss'].append(meta_loss)
        history['task_acc'].append(task_acc)
        
        print(f"Epoch {epoch+1} meta-train loss: {meta_loss:.4f}, task acc: {task_acc:.4f}")
        
        # Evaluate on validation set
        model.eval()
        val_running_corrects = 0
        val_total_samples = 0
        
        print("Evaluating on validation set...")
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                batch_size = inputs.size(0)
                val_total_samples += batch_size
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_acc = val_running_corrects.float() / val_total_samples
        history['val_acc'].append(val_epoch_acc.item())
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f'Val Acc: {val_epoch_acc:.4f}')
        
        # Save the best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_epoch_acc.item(),
                'num_classes': num_classes,
                'class_names': class_names
            }, os.path.join(model_dir, 'best_meta_model.pth'))
            
            print(f'New best model saved with accuracy: {val_epoch_acc:.4f}')
        
        # Save checkpoint every epoch
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_acc': val_epoch_acc.item(),
            'meta_loss': meta_loss,
            'task_acc': task_acc,
            'history': history
        }, os.path.join(model_dir, f'meta_checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'epochs_completed': config['meta_epochs'],
        'final_val_acc': val_epoch_acc.item(),
        'best_val_acc': best_val_acc.item(),
        'history': history,
        'num_classes': num_classes,
        'class_names': class_names
    }, os.path.join(model_dir, 'final_meta_model.pth'))
    
    print(f"Meta-learning completed. Final validation accuracy: {val_epoch_acc:.4f}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return model, history, train_loader, val_loader

# Utility function to find checkpoint or model file in a directory
def find_model_file(model_dir, preference='best_darts_model.pth'):
    """Find a model file in the directory, with preference given to specific filename"""
    if os.path.exists(os.path.join(model_dir, preference)):
        return os.path.join(model_dir, preference)
    
    for filename in os.listdir(model_dir):
        if filename.endswith('.pth'):
            print(f"Found model file: {filename}")
            return os.path.join(model_dir, filename)
    
    return None

# Modify meta_config to automatically find a model file if not explicitly specified
if not os.path.exists(meta_config['darts_model_path']):
    model_file = find_model_file(meta_config['model_dir'])
    if model_file:
        print(f"Using model file: {model_file}")
        meta_config['darts_model_path'] = model_file
    else:
        print("WARNING: No model file found. Will initialize from scratch.")

# Run the meta-learning - execute this cell to start training
try:
    model, history, train_loader, val_loader = run_meta_learning(meta_config)
    print("Meta-learning training complete!")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['meta_loss'])
    plt.title('Meta-Learning Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['task_acc'])
    plt.title('Task Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['val_acc'])
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(meta_config['model_dir'], 'meta_learning_history.png'))
    plt.show()
    
except Exception as e:
    print(f"Error during meta-learning: {e}")
    import traceback
    traceback.print_exc()

# Medical application inference function
def medical_inference(model_path, image_path, device='cuda', top_k=3):
    """
    Run inference on a medical image using the meta-learned model
    
    Args:
        model_path: Path to the saved model checkpoint
        image_path: Path to the medical image to classify
        device: Device to run inference on ('cuda' or 'cpu')
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with prediction results
    """
    from PIL import Image
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = ConvNeXtWithSearch(
        in_chans=3,
        num_classes=checkpoint.get('num_classes', 10),
        depths=checkpoint.get('depths', [3, 3, 9, 3]),
        dims=checkpoint.get('dims', [96, 192, 384, 768]),
        drop_path_rate=0.0
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Get class names
    class_names = checkpoint.get('class_names', [f"Class {i}" for i in range(model.num_classes)])
    
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess image
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)[0]
    
    # Get top-k predictions
    values, indices = torch.topk(probabilities, k=min(top_k, len(class_names)))
    
    # Format results
    predictions = []
    for i, (value, index) in enumerate(zip(values.cpu().numpy(), indices.cpu().numpy())):
        class_name = class_names[index]
        # Clean up class name if needed (remove count numbers, etc.)
        if '.' in class_name and ' ' in class_name:
            # Format like "1. Eczema 1677" -> "Eczema"
            cleaned_name = ' '.join(class_name.split(' ')[1:-1])
        else:
            cleaned_name = class_name
            
        predictions.append({
            'rank': i + 1,
            'class_id': int(index),
            'class_name': cleaned_name,
            'probability': float(value),
            'confidence': f"{float(value) * 100:.2f}%"
        })
    
    return {
        'image_path': image_path,
        'predictions': predictions,
        'model_path': model_path
    }

# Example of how to use the medical_inference function:
# results = medical_inference(
#     model_path='/mnt/Test/SC202/trained_models/best_meta_model.pth',
#     image_path='/path/to/your/medical_image.jpg',
#     device='cuda',
#     top_k=3
# )
# 
# print(f"Top predictions for {results['image_path']}:")
# for pred in results['predictions']:
#     print(f"{pred['rank']}. {pred['class_name']} - {pred['confidence']}")
