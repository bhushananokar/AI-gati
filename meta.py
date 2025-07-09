# Pretrained ConvNeXt Meta-Learning - Using pretrained weights for 90%+ accuracy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torchvision.models as models
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

# Configuration for Pretrained ConvNeXt Meta-Learning
pretrained_config = {
    'data_dir': '/mnt/Test/SC202/IMG_CLASSES',
    'model_dir': '/mnt/Test/SC202/trained_models',
    
    # Model parameters
    'img_size': 224,              # Keep standard ImageNet size for pretrained model
    'batch_size': 8,              # Small batches for stability
    'convnext_variant': 'tiny',   # Options: 'tiny', 'small', 'base', 'large'
    
    # Meta-learning parameters - optimized for pretrained features
    'meta_lr': 1e-4,              # Conservative for pretrained model
    'inner_lr': 0.01,             # Good adaptation rate
    'meta_epochs': 100,           
    'num_inner_steps': 5,         # Good adaptation with pretrained features
    'tasks_per_epoch': 12,        # Reasonable number of tasks
    'k_shot': 5,                  # 5-shot learning with good features
    'query_size': 5,              # Balanced query size
    'n_way': 3,                   # 3-way classification
    'first_order': False,         # Second-order for best accuracy
    
    # Training parameters
    'freeze_backbone': False,     # Allow backbone fine-tuning
    'gradient_clip': 1.0,         
    'warmup_epochs': 10,          # Moderate warmup
    'moving_avg_window': 7,       
    'early_stopping_patience': 20,
    'weight_decay': 1e-4,
    'dropout_rate': 0.1,
    
    'num_workers': 2,
    'use_cuda': True,
}

def load_data_pretrained(data_dir, img_size=224, batch_size=8, num_workers=2):
    """Load data with ImageNet-style preprocessing for pretrained ConvNeXt"""
    print(f"Loading dataset from {data_dir}")
    
    train_dir = os.path.join(data_dir, "train")
    
    if not os.path.exists(train_dir):
        raise ValueError("Cannot find train directory")
    
    # ImageNet-style preprocessing for pretrained ConvNeXt
    train_transform = transforms.Compose([
        transforms.Resize(256),                    # Standard ImageNet preprocessing
        transforms.CenterCrop(img_size),           # Keep center crop for consistency
        transforms.RandomHorizontalFlip(p=0.5),   # Standard augmentation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=num_workers, pin_memory=True)
    
    num_classes = len(dataset.classes)
    class_names = dataset.classes
    
    print(f"Dataset: {num_classes} classes, {len(dataset)} images")
    print(f"Classes: {class_names[:10]}...")
    
    return loader, num_classes, class_names

class PretrainedConvNeXtMeta(nn.Module):
    """Pretrained ConvNeXt adapted for meta-learning"""
    def __init__(self, num_classes=3, variant='tiny', freeze_backbone=False, dropout_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained ConvNeXt
        print(f"Loading pretrained ConvNeXt-{variant}...")
        if variant == 'tiny':
            self.backbone = models.convnext_tiny(pretrained=True)
        elif variant == 'small':
            self.backbone = models.convnext_small(pretrained=True)
        elif variant == 'base':
            self.backbone = models.convnext_base(pretrained=True)
        elif variant == 'large':
            self.backbone = models.convnext_large(pretrained=True)
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")
        
        # DYNAMIC FEATURE DETECTION: Run a dummy forward pass to get actual feature size
        print("🔍 Detecting actual feature dimensions...")
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            actual_feature_dim = dummy_features.shape[1]
        
        print(f"✅ Loaded ConvNeXt-{variant} with {actual_feature_dim}-dim features")
        
        # Remove the original classifier and replace with identity
        self.backbone.classifier = nn.Identity()
        
        # Verify the feature dimension again after removing classifier
        with torch.no_grad():
            dummy_features = self.backbone(dummy_input)
            if len(dummy_features.shape) > 2:
                # If features are not flattened, flatten them
                feature_dim = dummy_features.view(dummy_features.size(0), -1).shape[1]
                print(f"🔧 Features need flattening. Actual dimension: {feature_dim}")
                self.needs_flatten = True
            else:
                feature_dim = dummy_features.shape[1]
                self.needs_flatten = False
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("🔒 Freezing backbone parameters")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            print("🔓 Backbone parameters will be fine-tuned")
        
        # Create new classifier with correct input dimension
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize new classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        print(f"📊 Model created with {self.count_parameters():,} total parameters")
        print(f"📊 Feature dimension: {feature_dim}")
        if freeze_backbone:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"📊 Trainable parameters: {trainable:,} (classifier only)")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        # Extract features with pretrained backbone
        features = self.backbone(x)
        
        # Flatten if necessary
        if self.needs_flatten:
            features = features.view(features.size(0), -1)
        
        # Classify with new head
        output = self.classifier(features)
        return output
    
    def clone(self):
        """Create exact copy for inner loop"""
        clone = PretrainedConvNeXtMeta(
            num_classes=self.num_classes,
            variant=self.variant,
            freeze_backbone=self.freeze_backbone
        )
        clone.load_state_dict(self.state_dict())
        return clone

def create_pretrained_tasks(data_loader, num_tasks, n_way=3, k_shot=5, query_size=5):
    """Create tasks optimized for pretrained feature extraction"""
    print(f"Creating {num_tasks} tasks for pretrained ConvNeXt...")
    
    # Collect data by class with generous buffers
    class_data = {}
    sample_limit = k_shot + query_size + 15
    
    for inputs, labels in tqdm(data_loader, desc="Collecting data"):
        for i, label in enumerate(labels):
            label_item = label.item()
            if label_item not in class_data:
                class_data[label_item] = []
            
            if len(class_data[label_item]) < sample_limit:
                class_data[label_item].append(inputs[i].clone())
    
    # Filter classes with sufficient samples
    min_samples = k_shot + query_size + 2
    valid_classes = [c for c, data in class_data.items() if len(data) >= min_samples]
    
    print(f"Using {len(valid_classes)} classes with ≥{min_samples} samples each")
    
    if len(valid_classes) < n_way:
        raise ValueError(f"Need at least {n_way} classes, found {len(valid_classes)}")
    
    tasks = []
    successful_tasks = 0
    
    for task_idx in range(num_tasks * 2):  # Try more tasks in case some fail
        if successful_tasks >= num_tasks:
            break
            
        # Sample classes for this task
        try:
            task_classes = np.random.choice(valid_classes, size=n_way, replace=False)
        except ValueError:
            continue
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        task_valid = True
        
        for new_label, original_class in enumerate(task_classes):
            available_data = class_data[original_class]
            
            total_needed = k_shot + query_size
            if len(available_data) < total_needed:
                task_valid = False
                break
                
            # Sample without replacement
            indices = np.random.choice(len(available_data), size=total_needed, replace=False)
            
            # Support set
            for i in range(k_shot):
                support_data.append(available_data[indices[i]])
                support_labels.append(new_label)
            
            # Query set
            for i in range(k_shot, total_needed):
                query_data.append(available_data[indices[i]])
                query_labels.append(new_label)
        
        if task_valid and len(support_data) == n_way * k_shot and len(query_data) == n_way * query_size:
            support_data = torch.stack(support_data)
            support_labels = torch.tensor(support_labels, dtype=torch.long)
            query_data = torch.stack(query_data)
            query_labels = torch.tensor(query_labels, dtype=torch.long)
            
            tasks.append((support_data, support_labels, query_data, query_labels))
            successful_tasks += 1
    
    print(f"✅ Created {len(tasks)} valid tasks")
    return tasks

class PretrainedMAML:
    """MAML optimized for pretrained ConvNeXt"""
    def __init__(self, model, inner_lr=0.01, meta_lr=1e-4, num_inner_steps=5, 
                 gradient_clip=1.0, warmup_epochs=10):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.base_meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        
        # Separate learning rates for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # Different learning rates for pretrained vs new parameters
        self.meta_optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': meta_lr * 0.1},      # Lower LR for pretrained
            {'params': classifier_params, 'lr': meta_lr}           # Higher LR for new classifier
        ], weight_decay=1e-4)
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=100, eta_min=meta_lr/50
        )
        
        # Tracking
        self.loss_history = deque(maxlen=500)
        self.acc_history = deque(maxlen=500)
        
        print(f"🎯 MAML initialized with backbone LR: {meta_lr * 0.1:.1e}, classifier LR: {meta_lr:.1e}")
    
    def get_lr_scale(self, epoch):
        """Warmup scaling"""
        if epoch < self.warmup_epochs:
            return 0.1 + 0.9 * (epoch / self.warmup_epochs)
        return 1.0
    
    def inner_loop(self, support_data, support_labels, criterion, device):
        """Inner loop with pretrained features"""
        fast_model = self.model.clone().to(device)
        fast_model.train()
        
        inner_losses = []
        inner_accuracies = []
        
        for step in range(self.num_inner_steps):
            outputs = fast_model(support_data)
            loss = criterion(outputs, support_labels)
            inner_losses.append(loss.item())
            
            # Track adaptation
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                acc = (preds == support_labels).float().mean()
                inner_accuracies.append(acc.item())
            
            # Compute gradients
            gradients = torch.autograd.grad(
                loss, fast_model.parameters(),
                create_graph=True, retain_graph=False
            )
            
            # Apply gradients with different rates for backbone vs classifier
            with torch.no_grad():
                for (name, param), grad in zip(fast_model.named_parameters(), gradients):
                    if grad is not None:
                        # Lower learning rate for pretrained backbone
                        if 'classifier' in name:
                            lr = self.inner_lr
                        else:
                            lr = self.inner_lr * 0.1  # Much smaller for pretrained parts
                        
                        param.subtract_(lr * grad)
        
        return fast_model, inner_losses, inner_accuracies
    
    def meta_step(self, batch_tasks, criterion, device, epoch=0):
        """Meta step with pretrained model"""
        self.model.train()
        meta_losses = []
        task_accuracies = []
        inner_improvements = []
        
        lr_scale = self.get_lr_scale(epoch)
        
        print(f"  Processing {len(batch_tasks)} tasks with pretrained ConvNeXt...")
        
        for task_idx, (support_data, support_labels, query_data, query_labels) in enumerate(batch_tasks):
            try:
                support_data = support_data.to(device)
                support_labels = support_labels.to(device)
                query_data = query_data.to(device)
                query_labels = query_labels.to(device)
                
                # Inner loop adaptation
                fast_model, inner_losses, inner_accs = self.inner_loop(
                    support_data, support_labels, criterion, device
                )
                
                adaptation_improvement = inner_accs[-1] - inner_accs[0]
                inner_improvements.append(adaptation_improvement)
                
                # Query evaluation
                fast_model.eval()
                with torch.set_grad_enabled(True):
                    query_outputs = fast_model(query_data)
                    query_loss = criterion(query_outputs, query_labels)
                
                meta_losses.append(query_loss)
                
                # Calculate query accuracy
                with torch.no_grad():
                    _, preds = torch.max(query_outputs, 1)
                    accuracy = (preds == query_labels).float().mean()
                    task_accuracies.append(accuracy.item())
                
                if task_idx < 3:  # Show first few tasks
                    print(f"    Task {task_idx+1}: {inner_accs[0]:.3f} → {inner_accs[-1]:.3f} → Query: {accuracy:.3f}")
                
            except Exception as e:
                print(f"    Error in task {task_idx}: {e}")
                continue
        
        if len(meta_losses) > 0:
            meta_loss = torch.stack(meta_losses).mean()
            
            # Meta optimization with learning rate scaling
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Apply warmup scaling
            for param_group in self.meta_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale
            
            self.meta_optimizer.step()
            
            # Track metrics
            avg_accuracy = np.mean(task_accuracies)
            avg_improvement = np.mean(inner_improvements)
            
            self.loss_history.append(meta_loss.item())
            self.acc_history.append(avg_accuracy)
            
            return meta_loss.item(), avg_accuracy, avg_improvement
        
        return 0.0, 0.0, 0.0
    
    def get_smoothed_metrics(self, window=7):
        """Get smoothed metrics"""
        if len(self.acc_history) >= window:
            smooth_acc = np.mean(list(self.acc_history)[-window:])
            smooth_loss = np.mean(list(self.loss_history)[-window:])
            return smooth_loss, smooth_acc
        return 0.0, 0.0

def run_pretrained_convnext_meta_learning(config):
    """Meta-learning with pretrained ConvNeXt for 90%+ accuracy"""
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"🚀 PRETRAINED ConvNeXt Meta-Learning")
    print(f"Device: {device}")
    print(f"Target: 90%+ accuracy with pretrained features")
    
    # Load data
    train_loader, num_classes, class_names = load_data_pretrained(
        data_dir=config['data_dir'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Create pretrained ConvNeXt model
    model = PretrainedConvNeXtMeta(
        num_classes=config['n_way'],
        variant=config['convnext_variant'],
        freeze_backbone=config['freeze_backbone'],
        dropout_rate=config['dropout_rate']
    ).to(device)
    
    # Initialize MAML with pretrained model
    maml = PretrainedMAML(
        model=model,
        inner_lr=config['inner_lr'],
        meta_lr=config['meta_lr'],
        num_inner_steps=config['num_inner_steps'],
        gradient_clip=config['gradient_clip'],
        warmup_epochs=config['warmup_epochs']
    )
    
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    patience_counter = 0
    
    # Training history
    history = {
        'meta_loss': [],
        'task_acc': [],
        'smoothed_acc': [],
        'inner_improvement': []
    }
    
    print(f"\n📚 Training Configuration:")
    print(f"  Model: ConvNeXt-{config['convnext_variant']} (pretrained)")
    print(f"  {config['n_way']}-way, {config['k_shot']}-shot learning")
    print(f"  {config['num_inner_steps']} adaptation steps")
    print(f"  {config['tasks_per_epoch']} tasks per epoch")
    print(f"  Backbone frozen: {config['freeze_backbone']}")
    
    # Training loop
    for epoch in range(config['meta_epochs']):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{config['meta_epochs']}")
        print(f"{'='*60}")
        
        # Create tasks
        train_tasks = create_pretrained_tasks(
            train_loader,
            num_tasks=config['tasks_per_epoch'],
            n_way=config['n_way'],
            k_shot=config['k_shot'],
            query_size=config['query_size']
        )
        
        if len(train_tasks) == 0:
            print("❌ No valid tasks created, skipping epoch")
            continue
        
        # Meta training step
        meta_loss, task_acc, inner_improvement = maml.meta_step(
            train_tasks, criterion, device, epoch
        )
        
        # Get smoothed metrics
        smooth_loss, smooth_acc = maml.get_smoothed_metrics(
            window=config['moving_avg_window']
        )
        
        # Record history
        history['meta_loss'].append(meta_loss)
        history['task_acc'].append(task_acc)
        history['smoothed_acc'].append(smooth_acc)
        history['inner_improvement'].append(inner_improvement)
        
        # Step scheduler after warmup
        if epoch >= config['warmup_epochs']:
            maml.scheduler.step()
        
        current_backbone_lr = maml.meta_optimizer.param_groups[0]['lr']
        current_classifier_lr = maml.meta_optimizer.param_groups[1]['lr']
        epoch_time = time.time() - epoch_start
        
        # Rich logging
        print(f"\n📊 EPOCH {epoch+1} RESULTS ({epoch_time:.1f}s):")
        print(f"  Meta Loss: {meta_loss:.4f}")
        print(f"  Task Accuracy: {task_acc:.4f} ({task_acc*100:.1f}%)")
        print(f"  Smoothed Accuracy: {smooth_acc:.4f} ({smooth_acc*100:.1f}%)")
        print(f"  Inner Improvement: +{inner_improvement:.3f}")
        print(f"  LR - Backbone: {current_backbone_lr:.1e}, Classifier: {current_classifier_lr:.1e}")
        
        # Progress indicators
        if smooth_acc >= 0.95:
            print(f"  🎉 OUTSTANDING: {smooth_acc*100:.1f}% (≥95%)")
        elif smooth_acc >= 0.90:
            print(f"  🎯 TARGET ACHIEVED: {smooth_acc*100:.1f}% (≥90%)")
        elif smooth_acc >= 0.80:
            print(f"  ✅ EXCELLENT: {smooth_acc*100:.1f}% (≥80%)")
        elif smooth_acc >= 0.70:
            print(f"  📈 VERY GOOD: {smooth_acc*100:.1f}% (≥70%)")
        else:
            print(f"  🔄 LEARNING: {smooth_acc*100:.1f}% (<70%)")
        
        # Save best model
        if smooth_acc > best_accuracy:
            best_accuracy = smooth_acc
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'accuracy': smooth_acc,
                'config': config,
                'history': history
            }, os.path.join(config['model_dir'], 'best_pretrained_convnext_meta.pth'))
            
            print(f"  💾 NEW BEST: {smooth_acc*100:.1f}% (saved)")
        else:
            patience_counter += 1
            print(f"  ⏱️  No improvement: {patience_counter}/{config['early_stopping_patience']}")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n🛑 Early stopping after {patience_counter} epochs")
            break
        
        # Success check
        if smooth_acc >= 0.90:
            print(f"\n🎯 SUCCESS! Target 90%+ accuracy achieved: {smooth_acc*100:.1f}%")
        
        # Plot progress every 15 epochs
        if (epoch + 1) % 15 == 0:
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history['task_acc'], alpha=0.5, label='Raw Accuracy')
            plt.plot(history['smoothed_acc'], linewidth=2, label='Smoothed Accuracy')
            plt.axhline(y=0.9, color='g', linestyle='--', label='90% Target')
            plt.axhline(y=0.95, color='r', linestyle='--', label='95% Target')
            plt.title(f'ConvNeXt-{config["convnext_variant"]} Meta-Learning')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(history['meta_loss'])
            plt.title('Meta Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.plot(history['inner_improvement'])
            plt.title('Inner Loop Improvement')
            plt.ylabel('Accuracy Gain')
            plt.xlabel('Epoch')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(config['model_dir'], f'pretrained_convnext_progress_epoch_{epoch+1}.png'))
            plt.show()
    
    print(f"\n🏁 PRETRAINED ConvNeXt Meta-Learning COMPLETED!")
    print(f"🎯 Best accuracy achieved: {best_accuracy*100:.1f}%")
    
    if best_accuracy >= 0.95:
        print("🎉 OUTSTANDING: 95%+ accuracy with pretrained ConvNeXt!")
    elif best_accuracy >= 0.90:
        print("🎯 SUCCESS: 90%+ target achieved with pretrained features!")
    elif best_accuracy >= 0.80:
        print("✅ VERY GOOD: 80%+ accuracy - pretrained features working well")
    else:
        print("🔄 Consider: Different ConvNeXt variant or longer training")
    
    return model, history

# Run pretrained ConvNeXt meta-learning
print("🚀 Starting PRETRAINED ConvNeXt Meta-Learning")
print("Using ImageNet pretrained features for superior accuracy")
model, history = run_pretrained_convnext_meta_learning(pretrained_config)
