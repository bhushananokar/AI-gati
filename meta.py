# Medical ConvNeXt Meta-Learning - Optimized for Medical Image Prediction
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
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

# MEDICAL-OPTIMIZED Configuration for ConvNeXt Meta-Learning
medical_config = {
    'data_dir': '/mnt/Test/SC202/IMG_CLASSES',
    'model_dir': '/mnt/Test/SC202/trained_models',
    
    # Medical image parameters
    'img_size': 224,              # Standard for medical images
    'batch_size': 8,              # Conservative for medical data
    'convnext_variant': 'tiny',   # Good balance for medical
    
    # MEDICAL-OPTIMIZED meta-learning parameters
    'meta_lr': 5e-5,              # Very conservative for medical precision
    'inner_lr': 0.005,            # Careful adaptation for medical data
    'meta_epochs': 80,            # More epochs for medical accuracy
    'num_inner_steps': 7,         # More steps for medical precision
    'tasks_per_epoch': 8,         # Fewer, higher-quality tasks
    'k_shot': 8,                  # More examples for medical learning
    'query_size': 4,              # Smaller queries for stability
    'n_way': 2,                   # Binary medical classification (easier)
    'first_order': True,          # More stable for medical applications
    
    # Medical stability features
    'freeze_backbone': False,     # Fine-tune for medical domain
    'gradient_clip': 0.3,         # Very conservative clipping
    'warmup_epochs': 15,          # Longer warmup for stability
    'moving_avg_window': 10,      # Longer smoothing for medical
    'early_stopping_patience': 25, # More patience for medical accuracy
    'weight_decay': 1e-5,         # Light regularization
    'label_smoothing': 0.1,       # Uncertainty modeling for medical
    
    # Medical-specific augmentation
    'use_medical_augmentation': True,
    'color_jitter_strength': 0.05,  # Very light for medical images
    
    'num_workers': 2,
    'use_cuda': True,
}

def load_medical_data(data_dir, img_size=224, batch_size=8, num_workers=2, use_medical_augmentation=True):
    """Load data with medical-optimized preprocessing"""
    print(f"🏥 Loading MEDICAL dataset from {data_dir}")
    
    train_dir = os.path.join(data_dir, "train")
    
    if not os.path.exists(train_dir):
        raise ValueError("Cannot find train directory")
    
    if use_medical_augmentation:
        # MEDICAL-SPECIFIC augmentation - very conservative
        train_transform = transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.CenterCrop(img_size),              # Preserve important medical features
            transforms.RandomHorizontalFlip(p=0.3),      # Light flipping
            transforms.RandomRotation(5),                # Very small rotation
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),  # Minimal color changes
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Minimal augmentation for medical precision
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                       num_workers=num_workers, pin_memory=True)
    
    num_classes = len(dataset.classes)
    class_names = dataset.classes
    
    print(f"🏥 Medical Dataset: {num_classes} conditions, {len(dataset)} images")
    print(f"🏥 Medical Conditions: {class_names[:10]}...")
    
    # Medical dataset analysis
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    print(f"🏥 Dataset Balance Analysis:")
    for i, (class_name, count) in enumerate(zip(class_names[:5], [class_counts.get(i, 0) for i in range(5)])):
        print(f"   {class_name}: {count} samples")
    
    return loader, num_classes, class_names

class MedicalConvNeXt(nn.Module):
    """ConvNeXt optimized for medical image analysis"""
    def __init__(self, num_classes=2, variant='tiny', freeze_backbone=False, dropout_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.variant = variant
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained ConvNeXt
        print(f"🏥 Loading pretrained ConvNeXt-{variant} for MEDICAL analysis...")
        
        if variant == 'tiny':
            self.backbone = models.convnext_tiny(pretrained=True)
        elif variant == 'small':
            self.backbone = models.convnext_small(pretrained=True)
        elif variant == 'base':
            self.backbone = models.convnext_base(pretrained=True)
        else:
            raise ValueError(f"Unknown ConvNeXt variant: {variant}")
        
        # ROBUST feature dimension detection
        print("🔍 Detecting ConvNeXt feature dimensions...")
        
        # First, remove any existing classifier to get raw features
        original_classifier = self.backbone.classifier
        self.backbone.classifier = nn.Identity()
        
        # Test with actual medical image size
        self.backbone.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            
            print(f"🔧 Raw backbone output shape: {dummy_features.shape}")
            
            # Handle different output shapes
            if len(dummy_features.shape) == 4:  # (batch, channels, height, width)
                print("🔧 4D output detected - applying global average pooling")
                dummy_features = F.adaptive_avg_pool2d(dummy_features, (1, 1))
                dummy_features = dummy_features.view(dummy_features.size(0), -1)
                self.needs_pooling = True
            elif len(dummy_features.shape) == 3:  # (batch, seq_len, features)
                print("🔧 3D output detected - taking mean over sequence")
                dummy_features = dummy_features.mean(dim=1)
                self.needs_pooling = False
            elif len(dummy_features.shape) == 2:  # (batch, features)
                print("🔧 2D output detected - already flattened")
                self.needs_pooling = False
            else:
                print("🔧 Unknown output shape - flattening")
                dummy_features = dummy_features.view(dummy_features.size(0), -1)
                self.needs_pooling = False
            
            feature_dim = dummy_features.shape[1]
            print(f"✅ Final feature dimension: {feature_dim}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            print("🔒 Freezing backbone for medical domain adaptation")
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            print("🔓 Fine-tuning backbone for medical domain")
        
        # MEDICAL-SPECIFIC classifier with robust architecture
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize medical classifier conservatively
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"🏥 Medical ConvNeXt Model:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Feature dimension: {feature_dim}")
        print(f"   Needs pooling: {self.needs_pooling}")
    
    def forward(self, x):
        # Extract features with medical-optimized backbone
        features = self.backbone(x)
        
        # Handle different feature shapes robustly
        if self.needs_pooling:
            if len(features.shape) == 4:  # (batch, channels, height, width)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.view(features.size(0), -1)
            elif len(features.shape) == 3:  # (batch, seq_len, features)
                features = features.mean(dim=1)
            else:
                features = features.view(features.size(0), -1)
        elif len(features.shape) > 2:
            features = features.view(features.size(0), -1)
        
        # Medical classification
        output = self.classifier(features)
        return output
    
    def forward_with_uncertainty(self, x, num_samples=10):
        """Forward pass with uncertainty estimation for medical predictions"""
        self.train()  # Enable dropout for uncertainty
        outputs = []
        
        for _ in range(num_samples):
            output = self.forward(x)
            outputs.append(F.softmax(output, dim=1))
        
        outputs = torch.stack(outputs)
        mean_output = outputs.mean(dim=0)
        uncertainty = outputs.var(dim=0).sum(dim=1)  # Total uncertainty
        
        return mean_output, uncertainty
    
    def clone(self):
        """Create exact copy for medical meta-learning"""
        clone = MedicalConvNeXt(
            num_classes=self.num_classes,
            variant=self.variant,
            freeze_backbone=self.freeze_backbone
        )
        clone.load_state_dict(self.state_dict())
        return clone

def create_medical_tasks(data_loader, num_tasks, n_way=2, k_shot=8, query_size=4):
    """Create medical meta-learning tasks with quality control"""
    print(f"🏥 Creating {num_tasks} MEDICAL meta-learning tasks...")
    print(f"🏥 Task format: {n_way}-way, {k_shot}-shot medical classification")
    
    # Collect medical data with quality control
    class_data = {}
    sample_limit = k_shot + query_size + 15
    
    for inputs, labels in tqdm(data_loader, desc="Collecting medical data"):
        for i, label in enumerate(labels):
            label_item = label.item()
            if label_item not in class_data:
                class_data[label_item] = []
            
            if len(class_data[label_item]) < sample_limit:
                class_data[label_item].append(inputs[i].clone())
    
    # Quality control: only use classes with sufficient medical samples
    min_samples = k_shot + query_size + 3
    valid_classes = [c for c, data in class_data.items() if len(data) >= min_samples]
    
    print(f"🏥 Medical Quality Control:")
    print(f"   {len(valid_classes)} medical conditions with ≥{min_samples} samples")
    print(f"   Task requirement: {n_way} conditions per task")
    
    if len(valid_classes) < n_way:
        raise ValueError(f"Insufficient medical data: need ≥{n_way} conditions, found {len(valid_classes)}")
    
    # Create medical tasks with balanced sampling
    tasks = []
    
    for task_idx in range(num_tasks):
        # Sample medical conditions for this task
        task_classes = np.random.choice(valid_classes, size=n_way, replace=False)
        
        support_data, support_labels = [], []
        query_data, query_labels = [], []
        
        for new_label, original_class in enumerate(task_classes):
            available_data = class_data[original_class]
            
            total_needed = k_shot + query_size
            if len(available_data) >= total_needed:
                # Stratified sampling for medical data quality
                indices = np.random.choice(len(available_data), size=total_needed, replace=False)
                
                # Support set (training examples for each medical condition)
                for i in range(k_shot):
                    support_data.append(available_data[indices[i]])
                    support_labels.append(new_label)
                
                # Query set (test examples for each medical condition)
                for i in range(k_shot, total_needed):
                    query_data.append(available_data[indices[i]])
                    query_labels.append(new_label)
        
        # Create medical task with quality validation
        if len(support_data) == n_way * k_shot and len(query_data) == n_way * query_size:
            support_data = torch.stack(support_data)
            support_labels = torch.tensor(support_labels, dtype=torch.long)
            query_data = torch.stack(query_data)
            query_labels = torch.tensor(query_labels, dtype=torch.long)
            
            tasks.append((support_data, support_labels, query_data, query_labels))
    
    print(f"✅ Created {len(tasks)} high-quality medical tasks")
    print(f"🏥 Each task: {n_way} conditions × {k_shot} training + {query_size} test samples")
    
    return tasks

class MedicalMAML:
    """MAML specifically optimized for medical image analysis"""
    def __init__(self, model, inner_lr=0.005, meta_lr=5e-5, num_inner_steps=7, 
                 gradient_clip=0.3, warmup_epochs=15, weight_decay=1e-5, label_smoothing=0.1):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.base_meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        self.label_smoothing = label_smoothing
        
        # Medical-optimized optimizer with differential learning rates
        backbone_params = []
        classifier_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        
        # Conservative learning rates for medical precision
        self.meta_optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': meta_lr * 0.05, 'weight_decay': weight_decay},    # Very conservative for backbone
            {'params': classifier_params, 'lr': meta_lr, 'weight_decay': weight_decay}          # Standard for new classifier
        ])
        
        # Medical-friendly scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.meta_optimizer, T_max=80, eta_min=meta_lr/100
        )
        
        # Extended tracking for medical analysis
        self.loss_history = deque(maxlen=1000)
        self.acc_history = deque(maxlen=1000)
        self.uncertainty_history = deque(maxlen=1000)
        
        print(f"🏥 Medical MAML initialized:")
        print(f"   Backbone LR: {meta_lr * 0.05:.1e} (very conservative)")
        print(f"   Classifier LR: {meta_lr:.1e} (standard)")
        print(f"   Inner LR: {inner_lr} (medical-optimized)")
        print(f"   Label smoothing: {label_smoothing} (uncertainty modeling)")
    
    def get_lr_scale(self, epoch):
        """Extended warmup for medical stability"""
        if epoch < self.warmup_epochs:
            return 0.05 + 0.95 * (epoch / self.warmup_epochs)  # Very gradual medical warmup
        return 1.0
    
    def inner_loop(self, support_data, support_labels, criterion, device):
        """Medical-optimized inner loop adaptation"""
        fast_model = self.model.clone().to(device)
        fast_model.train()
        
        inner_losses = []
        inner_accuracies = []
        inner_confidences = []
        
        for step in range(self.num_inner_steps):
            outputs = fast_model(support_data)
            loss = criterion(outputs, support_labels)
            inner_losses.append(loss.item())
            
            # Track medical adaptation metrics
            with torch.no_grad():
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                acc = (preds == support_labels).float().mean()
                confidence = probs.max(dim=1)[0].mean()  # Average confidence
                
                inner_accuracies.append(acc.item())
                inner_confidences.append(confidence.item())
            
            # Compute gradients for medical adaptation
            gradients = torch.autograd.grad(
                loss, fast_model.parameters(),
                create_graph=True, retain_graph=False
            )
            
            # Conservative gradient application for medical precision
            with torch.no_grad():
                for param, grad in zip(fast_model.parameters(), gradients):
                    if grad is not None:
                        # Adaptive learning rate for medical stability
                        adaptive_lr = self.inner_lr
                        if step > 0 and inner_losses[-1] > inner_losses[-2]:
                            adaptive_lr *= 0.5  # Reduce if loss increases
                        
                        param.subtract_(adaptive_lr * grad)
        
        return fast_model, inner_losses, inner_accuracies, inner_confidences
    
    def meta_step(self, batch_tasks, criterion, device, epoch=0):
        """Medical-optimized meta step"""
        self.model.train()
        meta_losses = []
        task_accuracies = []
        task_confidences = []
        inner_improvements = []
        
        lr_scale = self.get_lr_scale(epoch)
        
        print(f"  🏥 Processing {len(batch_tasks)} medical tasks...")
        
        for task_idx, (support_data, support_labels, query_data, query_labels) in enumerate(batch_tasks):
            try:
                support_data = support_data.to(device)
                support_labels = support_labels.to(device)
                query_data = query_data.to(device)
                query_labels = query_labels.to(device)
                
                # Medical inner loop adaptation
                fast_model, inner_losses, inner_accs, inner_confs = self.inner_loop(
                    support_data, support_labels, criterion, device
                )
                
                adaptation_improvement = inner_accs[-1] - inner_accs[0]
                inner_improvements.append(adaptation_improvement)
                
                # Medical query evaluation
                fast_model.eval()
                with torch.set_grad_enabled(True):
                    query_outputs = fast_model(query_data)
                    query_loss = criterion(query_outputs, query_labels)
                
                meta_losses.append(query_loss)
                
                # Medical accuracy and confidence metrics
                with torch.no_grad():
                    query_probs = F.softmax(query_outputs, dim=1)
                    _, preds = torch.max(query_outputs, 1)
                    accuracy = (preds == query_labels).float().mean()
                    confidence = query_probs.max(dim=1)[0].mean()
                    
                    task_accuracies.append(accuracy.item())
                    task_confidences.append(confidence.item())
                
                if task_idx < 2:  # Show first few medical tasks
                    print(f"    🏥 Medical Task {task_idx+1}: {inner_accs[0]:.3f} → {inner_accs[-1]:.3f} → Query: {accuracy:.3f} (conf: {confidence:.3f})")
                
            except Exception as e:
                print(f"    ❌ Medical task {task_idx} error: {e}")
                continue
        
        if len(meta_losses) > 0:
            meta_loss = torch.stack(meta_losses).mean()
            
            # Medical meta optimization
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            
            # Conservative gradient clipping for medical stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            # Apply medical warmup scaling
            for param_group in self.meta_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * lr_scale
            
            self.meta_optimizer.step()
            
            # Track medical metrics
            avg_accuracy = np.mean(task_accuracies)
            avg_confidence = np.mean(task_confidences)
            avg_improvement = np.mean(inner_improvements)
            
            self.loss_history.append(meta_loss.item())
            self.acc_history.append(avg_accuracy)
            self.uncertainty_history.append(1.0 - avg_confidence)  # Uncertainty = 1 - confidence
            
            return meta_loss.item(), avg_accuracy, avg_improvement, avg_confidence
        
        return 0.0, 0.0, 0.0, 0.0
    
    def get_smoothed_metrics(self, window=10):
        """Medical-optimized smoothing"""
        if len(self.acc_history) > 0:
            actual_window = min(window, len(self.acc_history))
            smooth_acc = np.mean(list(self.acc_history)[-actual_window:])
            smooth_loss = np.mean(list(self.loss_history)[-actual_window:])
            smooth_uncertainty = np.mean(list(self.uncertainty_history)[-actual_window:]) if self.uncertainty_history else 0
            return smooth_loss, smooth_acc, smooth_uncertainty
        return 0.0, 0.0, 0.0

def run_medical_convnext_meta_learning(config):
    """Medical ConvNeXt Meta-Learning for 90%+ medical prediction accuracy"""
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"🏥 MEDICAL ConvNeXt Meta-Learning")
    print(f"🎯 Target: 90%+ accuracy for medical image prediction")
    print(f"⚕️  Optimized for medical domain with uncertainty estimation")
    print(f"Device: {device}")
    
    # Load medical data
    train_loader, num_classes, class_names = load_medical_data(
        data_dir=config['data_dir'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_medical_augmentation=config['use_medical_augmentation']
    )
    
    # Create medical ConvNeXt model
    model = MedicalConvNeXt(
        num_classes=config['n_way'],
        variant=config['convnext_variant'],
        freeze_backbone=config['freeze_backbone'],
        dropout_rate=0.2
    ).to(device)
    
    # Initialize medical MAML
    maml = MedicalMAML(
        model=model,
        inner_lr=config['inner_lr'],
        meta_lr=config['meta_lr'],
        num_inner_steps=config['num_inner_steps'],
        gradient_clip=config['gradient_clip'],
        warmup_epochs=config['warmup_epochs'],
        weight_decay=config['weight_decay'],
        label_smoothing=config['label_smoothing']
    )
    
    # Medical-optimized loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    best_accuracy = 0.0
    patience_counter = 0
    
    # Medical training history
    history = {
        'meta_loss': [],
        'task_acc': [],
        'smoothed_acc': [],
        'inner_improvement': [],
        'medical_confidence': [],
        'medical_uncertainty': []
    }
    
    print(f"\n🏥 Medical Meta-Learning Configuration:")
    print(f"   Model: ConvNeXt-{config['convnext_variant']} (medical-optimized)")
    print(f"   Medical task format: {config['n_way']}-way, {config['k_shot']}-shot")
    print(f"   Inner adaptation steps: {config['num_inner_steps']}")
    print(f"   Medical tasks per epoch: {config['tasks_per_epoch']}")
    print(f"   Label smoothing: {config['label_smoothing']} (uncertainty modeling)")
    
    # Medical training loop
    for epoch in range(config['meta_epochs']):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"MEDICAL EPOCH {epoch+1}/{config['meta_epochs']} - ConvNeXt Meta-Learning")
        print(f"{'='*70}")
        
        # Create medical tasks
        train_tasks = create_medical_tasks(
            train_loader,
            num_tasks=config['tasks_per_epoch'],
            n_way=config['n_way'],
            k_shot=config['k_shot'],
            query_size=config['query_size']
        )
        
        if len(train_tasks) == 0:
            print("❌ No valid medical tasks created")
            continue
        
        # Medical meta training step
        meta_loss, task_acc, inner_improvement, confidence = maml.meta_step(
            train_tasks, criterion, device, epoch
        )
        
        # Get medical smoothed metrics
        smooth_loss, smooth_acc, smooth_uncertainty = maml.get_smoothed_metrics(
            window=config['moving_avg_window']
        )
        
        # Record medical history
        history['meta_loss'].append(meta_loss)
        history['task_acc'].append(task_acc)
        history['smoothed_acc'].append(smooth_acc)
        history['inner_improvement'].append(inner_improvement)
        history['medical_confidence'].append(confidence)
        history['medical_uncertainty'].append(smooth_uncertainty)
        
        # Step medical scheduler after warmup
        if epoch >= config['warmup_epochs']:
            maml.scheduler.step()
        
        current_backbone_lr = maml.meta_optimizer.param_groups[0]['lr']
        current_classifier_lr = maml.meta_optimizer.param_groups[1]['lr']
        epoch_time = time.time() - epoch_start
        
        # Rich medical logging
        print(f"\n🏥 MEDICAL EPOCH {epoch+1} RESULTS ({epoch_time:.1f}s):")
        print(f"  Meta Loss: {meta_loss:.4f}")
        print(f"  Task Accuracy: {task_acc:.4f} ({task_acc*100:.1f}%)")
        print(f"  Smoothed Accuracy: {smooth_acc:.4f} ({smooth_acc*100:.1f}%)")
        print(f"  Inner Improvement: +{inner_improvement:.3f}")
        print(f"  Medical Confidence: {confidence:.3f}")
        print(f"  Medical Uncertainty: {smooth_uncertainty:.3f}")
        print(f"  LR - Backbone: {current_backbone_lr:.1e}, Classifier: {current_classifier_lr:.1e}")
        
        # Medical progress indicators
        current_accuracy = smooth_acc if smooth_acc > 0 else task_acc
        
        if current_accuracy >= 0.95:
            print(f"  🎉 OUTSTANDING MEDICAL ACCURACY: {current_accuracy*100:.1f}% (≥95%)")
        elif current_accuracy >= 0.90:
            print(f"  🏥 EXCELLENT MEDICAL ACCURACY: {current_accuracy*100:.1f}% (≥90%)")
        elif current_accuracy >= 0.85:
            print(f"  ✅ VERY GOOD MEDICAL ACCURACY: {current_accuracy*100:.1f}% (≥85%)")
        elif current_accuracy >= 0.80:
            print(f"  📈 GOOD MEDICAL ACCURACY: {current_accuracy*100:.1f}% (≥80%)")
        elif current_accuracy >= 0.70:
            print(f"  📊 MODERATE MEDICAL ACCURACY: {current_accuracy*100:.1f}% (≥70%)")
        else:
            print(f"  🔄 MEDICAL LEARNING IN PROGRESS: {current_accuracy*100:.1f}% (<70%)")
        
        # Medical adaptation quality
        if inner_improvement > 0.15:
            print(f"  🔥 EXCELLENT MEDICAL ADAPTATION: Strong learning from medical examples")
        elif inner_improvement > 0.08:
            print(f"  ✅ GOOD MEDICAL ADAPTATION: Solid learning from medical data")
        elif inner_improvement > 0.04:
            print(f"  📈 MODERATE MEDICAL ADAPTATION: Gradual medical learning")
        else:
            print(f"  ⚠️  WEAK MEDICAL ADAPTATION: May need parameter tuning")
        
        # Medical confidence analysis
        if confidence > 0.8:
            print(f"  🎯 HIGH MEDICAL CONFIDENCE: Model is confident in predictions")
        elif confidence > 0.65:
            print(f"  ✅ GOOD MEDICAL CONFIDENCE: Reasonable certainty")
        else:
            print(f"  ⚠️  LOW MEDICAL CONFIDENCE: High uncertainty in predictions")
        
        # Save best medical model
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'accuracy': current_accuracy,
                'confidence': confidence,
                '
