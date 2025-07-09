# State-of-the-Art Models for Dermatological Disease Classification
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

# DERMATOLOGY-OPTIMIZED Configuration
dermatology_config = {
    'data_dir': '/mnt/Test/SC202/IMG_CLASSES',
    'model_dir': '/mnt/Test/SC202/trained_models',
    
    # Model selection - try different ones!
    'model_type': 'efficientnet_b3',  # Options: 'efficientnet_b3', 'vit_b_16', 'swin_b', 'maxvit_t', 'densenet121'
    'img_size': 224,                  # 384 for ViT/Swin for better results
    'batch_size': 16,                 # Adjust based on model size
    
    # DERMATOLOGY-OPTIMIZED training
    'learning_rate': 1e-4,            # Conservative for medical
    'epochs': 100,                    # More epochs for medical precision
    'weight_decay': 1e-4,
    'dropout_rate': 0.3,
    'label_smoothing': 0.1,           # Good for dermatology uncertainty
    
    # Dermatology-specific features
    'use_advanced_augmentation': True,
    'use_mixup': False,               # Can help with limited data
    'use_cutmix': False,              # Can help with limited data
    'use_test_time_augmentation': True,
    
    # Training optimization
    'use_scheduler': True,
    'scheduler_type': 'cosine',       # 'step', 'cosine', 'plateau'
    'early_stopping_patience': 15,
    'validation_split': 0.2,
    'save_best_model': True,
    
    'num_workers': 4,
    'use_cuda': True,
}

def load_dermatology_data(data_dir, img_size=224, batch_size=16, validation_split=0.2, 
                         use_advanced_augmentation=True, num_workers=4):
    """Load dermatology data with specialized preprocessing"""
    print(f"🔬 Loading DERMATOLOGY dataset from {data_dir}")
    
    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(train_dir):
        raise ValueError("Cannot find train directory")
    
    # DERMATOLOGY-SPECIFIC augmentation
    if use_advanced_augmentation:
        # Advanced augmentation for skin lesions
        train_transform = transforms.Compose([
            transforms.Resize((img_size + 56, img_size + 56)),
            transforms.RandomCrop(img_size),
            
            # Dermatology-specific augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),        # Skin lesions can be flipped
            transforms.RandomRotation(30),               # Lesions can be at any angle
            
            # Color augmentations - important for skin tones
            transforms.ColorJitter(
                brightness=0.15,     # Lighting variations
                contrast=0.15,       # Different skin contrasts
                saturation=0.15,     # Skin tone variations
                hue=0.05            # Slight hue changes
            ),
            
            # Advanced geometric transforms
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),    # Small translations
                scale=(0.9, 1.1),        # Slight scaling
                shear=5                   # Small shear for perspective
            ),
            
            # Simulate different dermatoscope conditions
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.2),
            
            transforms.ToTensor(),
            
            # ImageNet normalization (good for pretrained models)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # Additional dermatology-specific augmentation
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Simulate hair/artifacts
        ])
    else:
        # Basic augmentation
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    full_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Apply validation transform to validation set
    val_dataset.dataset = datasets.ImageFolder(root=train_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes
    
    print(f"🔬 Dermatology Dataset:")
    print(f"   {num_classes} skin conditions")
    print(f"   {train_size} training images")
    print(f"   {val_size} validation images")
    print(f"   Conditions: {class_names}")
    
    return train_loader, val_loader, num_classes, class_names

class DermatologyClassifier(nn.Module):
    """State-of-the-art classifier for dermatological diseases"""
    def __init__(self, num_classes, model_type='efficientnet_b3', dropout_rate=0.3, 
                 img_size=224, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.model_type = model_type
        self.img_size = img_size
        
        print(f"🔬 Creating {model_type} for dermatology classification...")
        
        # Load state-of-the-art models
        if model_type == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif model_type == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = 1536
        elif model_type == 'efficientnet_b7':
            self.backbone = models.efficientnet_b7(pretrained=pretrained)
            feature_dim = 2560
        elif model_type == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)
            feature_dim = 768
        elif model_type == 'vit_l_16':
            self.backbone = models.vit_l_16(pretrained=pretrained)
            feature_dim = 1024
        elif model_type == 'swin_t':
            self.backbone = models.swin_t(pretrained=pretrained)
            feature_dim = 768
        elif model_type == 'swin_b':
            self.backbone = models.swin_b(pretrained=pretrained)
            feature_dim = 1024
        elif model_type == 'maxvit_t':
            self.backbone = models.maxvit_t(pretrained=pretrained)
            feature_dim = 512
        elif model_type == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            feature_dim = 1024
        elif model_type == 'regnet_y_32gf':
            self.backbone = models.regnet_y_32gf(pretrained=pretrained)
            feature_dim = 3712
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Remove original classifier
        if hasattr(self.backbone, 'classifier'):
            if isinstance(self.backbone.classifier, nn.Sequential):
                feature_dim = self.backbone.classifier[-1].in_features
            else:
                feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'fc'):
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif hasattr(self.backbone, 'heads'):
            feature_dim = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()
        
        # Dermatology-optimized classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1) if 'efficientnet' in model_type or 'densenet' in model_type else nn.Identity(),
            nn.Flatten() if 'efficientnet' in model_type or 'densenet' in model_type else nn.Identity(),
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
        
        # Initialize classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ {model_type} created: {total_params:,} parameters")
        print(f"   Feature dimension: {feature_dim}")
        print(f"   Image size: {img_size}x{img_size}")
    
    def forward(self, x):
        features = self.backbone(x)
        
        # Handle different output formats
        if len(features.shape) > 2:
            # For models that output feature maps
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        
        return self.classifier(features)
    
    def forward_with_tta(self, x, num_augmentations=5):
        """Test Time Augmentation for better dermatology predictions"""
        self.eval()
        predictions = []
        
        # Original image
        with torch.no_grad():
            pred = F.softmax(self.forward(x), dim=1)
            predictions.append(pred)
        
        # Augmented versions
        for _ in range(num_augmentations):
            # Random augmentations
            augmented = x.clone()
            
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                augmented = torch.flip(augmented, dims=[3])
            
            # Random vertical flip
            if torch.rand(1) > 0.5:
                augmented = torch.flip(augmented, dims=[2])
            
            with torch.no_grad():
                pred = F.softmax(self.forward(augmented), dim=1)
                predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)

def train_dermatology_classifier(config):
    """Train state-of-the-art dermatology classifier"""
    device = torch.device('cuda' if config['use_cuda'] and torch.cuda.is_available() else 'cpu')
    print(f"🚀 DERMATOLOGY Classification with {config['model_type'].upper()}")
    print(f"🔬 State-of-the-art approach for skin disease classification")
    print(f"🎯 Target: 90%+ accuracy for dermatological diseases")
    print(f"Device: {device}")
    
    # Load dermatology data
    train_loader, val_loader, num_classes, class_names = load_dermatology_data(
        data_dir=config['data_dir'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        validation_split=config['validation_split'],
        use_advanced_augmentation=config['use_advanced_augmentation'],
        num_workers=config['num_workers']
    )
    
    # Create state-of-the-art model
    model = DermatologyClassifier(
        num_classes=num_classes,
        model_type=config['model_type'],
        dropout_rate=config['dropout_rate'],
        img_size=config['img_size']
    ).to(device)
    
    # Dermatology-optimized loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    # Different optimizers for different models
    if 'vit' in config['model_type'] or 'swin' in config['model_type']:
        # Transformers prefer AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        # CNNs work well with Adam
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    
    # Learning rate scheduler
    if config['use_scheduler']:
        if config['scheduler_type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs'], eta_min=config['learning_rate']/100
            )
        elif config['scheduler_type'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=20, gamma=0.5
            )
        else:  # plateau
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=7, verbose=True
            )
    
    # Training tracking
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\n🔬 Dermatology Training Configuration:")
    print(f"   Model: {config['model_type']} (state-of-the-art)")
    print(f"   Skin conditions: {num_classes}")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Image size: {config['img_size']}x{config['img_size']}")
    print(f"   Advanced augmentation: {config['use_advanced_augmentation']}")
    
    # Training loop for dermatology
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{config['epochs']} - {config['model_type'].upper()} Dermatology")
        print(f"{'='*70}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training {config['model_type']}")
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        train_accuracy = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_confidences = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation {config['model_type']}")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Use TTA if enabled
                if config.get('use_test_time_augmentation', False):
                    outputs = model.forward_with_tta(inputs)
                else:
                    outputs = model(inputs)
                    outputs = F.softmax(outputs, dim=1)
                
                # Calculate loss (need logits for loss)
                if config.get('use_test_time_augmentation', False):
                    loss = criterion(torch.log(outputs + 1e-8), labels)
                else:
                    loss = criterion(model(inputs), labels)
                
                val_loss += loss.item()
                confidence, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Track confidence
                all_confidences.extend(confidence.cpu().numpy())
                
                # Update progress
                current_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        val_accuracy = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        avg_confidence = np.mean(all_confidences)
        
        # Record metrics
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # Learning rate scheduling
        if config['use_scheduler']:
            if config['scheduler_type'] == 'plateau':
                scheduler.step(val_accuracy)
            else:
                scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        
        # Rich logging for dermatology
        print(f"\n🔬 {config['model_type'].upper()} EPOCH {epoch+1} RESULTS ({epoch_time:.1f}s):")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f} ({train_accuracy*100:.1f}%)")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f} ({val_accuracy*100:.1f}%)")
        print(f"  Confidence: {avg_confidence:.3f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Dermatology progress indicators
        if val_accuracy >= 0.95:
            print(f"  🎉 OUTSTANDING DERMATOLOGY: {val_accuracy*100:.1f}% (≥95%)")
        elif val_accuracy >= 0.90:
            print(f"  🔬 EXCELLENT DERMATOLOGY: {val_accuracy*100:.1f}% (≥90%)")
        elif val_accuracy >= 0.85:
            print(f"  ✅ VERY GOOD DERMATOLOGY: {val_accuracy*100:.1f}% (≥85%)")
        elif val_accuracy >= 0.80:
            print(f"  📈 GOOD DERMATOLOGY: {val_accuracy*100:.1f}% (≥80%)")
        elif val_accuracy >= 0.75:
            print(f"  📊 DECENT DERMATOLOGY: {val_accuracy*100:.1f}% (≥75%)")
        else:
            print(f"  🔄 LEARNING: {val_accuracy*100:.1f}% (<75%)")
        
        # Model-specific feedback
        if 'efficientnet' in config['model_type']:
            print(f"  ⚡ EfficientNet: Optimized for medical imaging efficiency")
        elif 'vit' in config['model_type']:
            print(f"  👁️ Vision Transformer: Attention-based lesion analysis")
        elif 'swin' in config['model_type']:
            print(f"  🪟 Swin Transformer: Hierarchical vision processing")
        elif 'maxvit' in config['model_type']:
            print(f"  🚀 MaxViT: Cutting-edge CNN+Transformer hybrid")
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            
            if config['save_best_model']:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'train_accuracy': train_accuracy,
                    'val_accuracy': val_accuracy,
                    'confidence': avg_confidence,
                    'config': config,
                    'class_names': class_names,
                    'model_type': f'dermatology_{config["model_type"]}'
                }, os.path.join(config['model_dir'], f'best_dermatology_{config["model_type"]}.pth'))
            
            print(f"  💾 NEW BEST {config['model_type'].upper()}: {val_accuracy*100:.1f}% (saved)")
        else:
            patience_counter += 1
            print(f"  ⏱️  No improvement: {patience_counter}/{config['early_stopping_patience']}")
        
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n🛑 Early stopping after {patience_counter} epochs without improvement")
            break
        
        # Success check
        if val_accuracy >= 0.90:
            print(f"\n🎯 DERMATOLOGY SUCCESS! 90%+ accuracy: {val_accuracy*100:.1f}%")
            print(f"🔬 {config['model_type']} ready for dermatological deployment!")
        
        # Plot progress every 15 epochs
        if (epoch + 1) % 15 == 0:
            plt.figure(figsize=(15, 5))
            
            # Accuracy comparison
            plt.subplot(1, 3, 1)
            epochs = range(1, len(train_accuracies) + 1)
            plt.plot(epochs, [acc*100 for acc in train_accuracies], 'b-', label=f'Train ({config["model_type"]})')
            plt.plot(epochs, [acc*100 for acc in val_accuracies], 'r-', label=f'Val ({config["model_type"]})')
            plt.axhline(y=90, color='g', linestyle='--', label='90% Target')
            plt.title(f'{config["model_type"].upper()} Dermatology Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)
            
            # Loss plot
            plt.subplot(1, 3, 2)
            plt.plot(epochs, train_losses, 'b-', label='Train Loss')
            plt.plot(epochs, val_losses, 'r-', label='Val Loss')
            plt.title(f'{config["model_type"].upper()} Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # Performance evolution
            plt.subplot(1, 3, 3)
            plt.plot(epochs, [acc*100 for acc in val_accuracies], 'r-', linewidth=2)
            plt.fill_between(epochs, [acc*100 for acc in val_accuracies], alpha=0.3, color='red')
            plt.axhline(y=90, color='g', linestyle='--', alpha=0.7)
            plt.title(f'{config["model_type"].upper()} Progress')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Accuracy (%)')
            plt.grid(True)
            
            plt.suptitle(f'Dermatology {config["model_type"].upper()} Training - Epoch {epoch+1}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(config['model_dir'], f'dermatology_{config["model_type"]}_epoch_{epoch+1}.png'))
            plt.show()
    
    print(f"\n🔬 DERMATOLOGY {config['model_type'].upper()} COMPLETED!")
    print(f"🎯 Best validation accuracy: {best_val_acc*100:.1f}%")
    
    # Final performance summary
    if best_val_acc >= 0.95:
        print(f"🎉 OUTSTANDING: 95%+ accuracy with {config['model_type']}!")
    elif best_val_acc >= 0.90:
        print(f"🔬 EXCELLENT: 90%+ dermatology accuracy with {config['model_type']}!")
    elif best_val_acc >= 0.85:
        print(f"✅ VERY GOOD: 85%+ accuracy - {config['model_type']} performing well!")
    elif best_val_acc >= 0.80:
        print(f"📈 GOOD: 80%+ accuracy with {config['model_type']}")
    else:
        print(f"🔄 LEARNING: {config['model_type']} needs more training")
    
    print(f"\n🔬 Dermatology Model Summary:")
    print(f"   Architecture: {config['model_type']} (state-of-the-art)")
    print(f"   Best accuracy: {best_val_acc*100:.1f}%")
    print(f"   Skin conditions: {num_classes}")
    print(f"   Training epochs: {epoch+1}")
    print(f"   Ready for deployment: {'YES' if best_val_acc >= 0.85 else 'CONTINUE TRAINING'}")
    
    return model, {
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_accuracy': best_val_acc,
        'model_type': config['model_type']
    }

# Dermatology inference with state-of-the-art models
def dermatology_inference(model_path, image_path, device='cuda', use_tta=True):
    """
    Advanced dermatology inference with state-of-the-art models
    
    Args:
        model_path: Path to trained dermatology model
        image_path: Path to skin lesion image
        device: Device for inference
        use_tta: Use Test Time Augmentation for better accuracy
    
    Returns:
        Dictionary with dermatology predictions
    """
    from PIL import Image
    
    print(f"🔬 Loading dermatology model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model info
    config = checkpoint.get('config', {})
    class_names = checkpoint.get('class_names', [])
    num_classes = len(class_names)
    model_type = config.get('model_type', 'efficientnet_b3')
    
    print(f"🔬 Model: {model_type} for {num_classes} skin conditions")
    
    # Create model
    model = DermatologyClassifier(
        num_classes=num_classes,
        model_type=model_type,
        dropout_rate=config.get('dropout_rate', 0.3),
        img_size=config.get('img_size', 224)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess dermatology image
    print(f"🔬 Processing dermatology image: {image_path}")
    img = Image.open(image_path).convert('RGB')
    
    # Dermatology-specific preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.get('img_size', 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Advanced dermatology inference
    print(f"🔬 Running {model_type} inference...")
    
    with torch.no_grad():
        if use_tta:
            # Test Time Augmentation for better dermatology accuracy
            probabilities = model.forward_with_tta(img_tensor, num_augmentations=8)[0]
        else:
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        confidence, predicted = torch.max(probabilities, 0)
    
    # Format dermatology results
    results = []
    for i, (condition, prob) in enumerate(zip(class_names, probabilities)):
        # Clean condition names
        clean_condition = condition
        if '.' in condition and condition.split('.')[0].isdigit():
            clean_condition = ' '.join(condition.split('.')[1:]).strip()
            if clean_condition.split()[-1].isdigit():
                clean_condition = ' '.join(clean_condition.split()[:-1])
        
        results.append({
            'condition': clean_condition,
            'probability': float(prob),
            'confidence_percentage': f"{float(prob)*100:.1f}%"
        })
    
    # Sort by probability
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    # Dermatology-specific interpretation
    top_prob = results[0]['probability']
    if top_prob > 0.9:
        interpretation = f"High confidence {model_type} diagnosis"
    elif top_prob > 0.8:
        interpretation = f"Good confidence {model_type} diagnosis"
    elif top_prob > 0.7:
        interpretation = f"Moderate confidence {model_type} diagnosis"
    else:
        interpretation = f"Low confidence {model_type} diagnosis - recommend dermatologist review"
    
    return {
        'image': image_path,
        'top_diagnosis': results[0],
        'all_diagnoses': results,
        'model_confidence': float(confidence),
        'clinical_interpretation': interpretation,
        'model_info': {
            'architecture': model_type,
            'accuracy': f"{checkpoint.get('val_accuracy', 0)*100:.1f}%",
            'skin_conditions': len(class_names),
            'test_time_augmentation': use_tta
        },
        'dermatology_notes': {
            'recommendation': 'AI-assisted diagnosis - always consult dermatologist for clinical decisions',
            'confidence_levels': 'High: >90%, Good: 80-90%, Moderate: 70-80%, Low: <70%',
            'model_strength': f'{model_type} optimized for dermatological image analysis'
        }
    }

# Model comparison function
def compare_dermatology_models(data_dir, model_types=['efficientnet_b3', 'vit_b_16', 'swin_b'], 
                              epochs_per_model=20, img_size=224):
    """
    Compare different state-of-the-art models for dermatology
    
    Args:
        data_dir: Path to dermatology dataset
        model_types: List of models to compare
        epochs_per_model: Training epochs per model
        img_size: Image size for training
    
    Returns:
        Dictionary with comparison results
    """
    print(f"🔬 DERMATOLOGY MODEL COMPARISON")
    print(f"Models to compare: {model_types}")
    print(f"Training epochs per model: {epochs_per_model}")
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"TRAINING {model_type.upper()} FOR DERMATOLOGY")
        print(f"{'='*70}")
        
        # Configure for this model
        config = dermatology_config.copy()
        config.update({
            'model_type': model_type,
            'epochs': epochs_per_model,
            'img_size': img_size,
            'batch_size': 16 if 'efficientnet_b7' not in model_type else 8,  # Adjust for model size
        })
        
        try:
            # Train model
            model, history = train_dermatology_classifier(config)
            
            # Store results
            results[model_type] = {
                'best_accuracy': history['best_accuracy'],
                'final_train_acc': history['train_accuracies'][-1],
                'final_val_acc': history['val_accuracies'][-1],
                'training_history': history,
                'epochs_trained': len(history['train_accuracies']),
                'model_size': sum(p.numel() for p in model.parameters()),
            }
            
            print(f"✅ {model_type} completed: {history['best_accuracy']*100:.1f}% best accuracy")
            
        except Exception as e:
            print(f"❌ {model_type} failed: {e}")
            results[model_type] = {'error': str(e)}
    
    # Create comparison visualization
    plt.figure(figsize=(20, 10))
    
    # Best accuracy comparison
    plt.subplot(2, 3, 1)
    models = [m for m in model_types if m in results and 'best_accuracy' in results[m]]
    accuracies = [results[m]['best_accuracy']*100 for m in models]
    colors = ['blue', 'red', 'green', 'orange', 'purple'][:len(models)]
    
    bars = plt.bar(models, accuracies, color=colors)
    plt.title('Best Validation Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.axhline(y=90, color='black', linestyle='--', label='90% Target')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Model size comparison
    plt.subplot(2, 3, 2)
    sizes = [results[m]['model_size']/1e6 for m in models]  # Convert to millions
    bars = plt.bar(models, sizes, color=colors)
    plt.title('Model Size Comparison')
    plt.ylabel('Parameters (Millions)')
    plt.xticks(rotation=45)
    
    for bar, size in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{size:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Training curves comparison
    plt.subplot(2, 3, 3)
    for i, model_type in enumerate(models):
        if 'training_history' in results[model_type]:
            history = results[model_type]['training_history']
            epochs = range(1, len(history['val_accuracies']) + 1)
            plt.plot(epochs, [acc*100 for acc in history['val_accuracies']], 
                    color=colors[i], label=model_type, linewidth=2)
    
    plt.title('Validation Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=90, color='black', linestyle='--', alpha=0.5)
    
    # Efficiency analysis (accuracy per parameter)
    plt.subplot(2, 3, 4)
    efficiency = [accuracies[i] / sizes[i] for i in range(len(models))]
    bars = plt.bar(models, efficiency, color=colors)
    plt.title('Efficiency (Accuracy per Million Parameters)')
    plt.ylabel('Accuracy % / Million Params')
    plt.xticks(rotation=45)
    
    for bar, eff in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    
    # Summary table
    plt.subplot(2, 3, 5)
    plt.axis('off')
    
    # Create summary text
    summary_text = "🔬 DERMATOLOGY MODEL COMPARISON SUMMARY\n\n"
    
    # Sort by accuracy
    sorted_models = sorted([(m, results[m]['best_accuracy']) for m in models], 
                          key=lambda x: x[1], reverse=True)
    
    for i, (model, acc) in enumerate(sorted_models):
        rank = i + 1
        size = results[model]['model_size'] / 1e6
        eff = acc * 100 / size
        
        summary_text += f"{rank}. {model.upper()}\n"
        summary_text += f"   Accuracy: {acc*100:.1f}%\n"
        summary_text += f"   Size: {size:.1f}M params\n"
        summary_text += f"   Efficiency: {eff:.2f}\n\n"
    
    # Add recommendations
    best_accuracy = sorted_models[0]
    best_efficiency = max(models, key=lambda m: results[m]['best_accuracy']*100 / (results[m]['model_size']/1e6))
    
    summary_text += "🎯 RECOMMENDATIONS:\n"
    summary_text += f"Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]*100:.1f}%)\n"
    summary_text += f"Best Efficiency: {best_efficiency}\n"
    summary_text += f"For Production: {'efficientnet_b3' if 'efficientnet_b3' in models else models[0]}"
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Performance vs Size scatter
    plt.subplot(2, 3, 6)
    plt.scatter(sizes, accuracies, c=colors, s=100, alpha=0.7)
    
    for i, model in enumerate(models):
        plt.annotate(model, (sizes[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.xlabel('Model Size (Million Parameters)')
    plt.ylabel('Best Accuracy (%)')
    plt.title('Accuracy vs Model Size')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Dermatology Model Comparison - State-of-the-Art Analysis', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(dermatology_config['model_dir'], 'dermatology_model_comparison.png'), dpi=150)
    plt.show()
    
    return results

# Example usage and recommendations
def run_dermatology_sota_example():
    """Example of how to use state-of-the-art models for dermatology"""
    
    print("🔬 DERMATOLOGY STATE-OF-THE-ART MODEL EXAMPLES")
    print("=" * 60)
    
    # Example 1: Train EfficientNet-B3 (recommended)
    print("\n1. Training EfficientNet-B3 (RECOMMENDED for dermatology):")
    """
    config = dermatology_config.copy()
    config['model_type'] = 'efficientnet_b3'
    model, history = train_dermatology_classifier(config)
    """
    
    # Example 2: Train Vision Transformer
    print("\n2. Training Vision Transformer (good for attention analysis):")
    """
    config = dermatology_config.copy()
    config.update({
        'model_type': 'vit_b_16',
        'img_size': 384,  # Larger images for ViT
        'batch_size': 8,  # Smaller batch for larger images
        'learning_rate': 5e-5  # Lower LR for transformer
    })
    model, history = train_dermatology_classifier(config)
    """
    
    # Example 3: Model comparison
    print("\n3. Compare multiple state-of-the-art models:")
    """
    results = compare_dermatology_models(
        data_dir='/mnt/Test/SC202/IMG_CLASSES',
        model_types=['efficientnet_b3', 'vit_b_16', 'swin_b'],
        epochs_per_model=20
    )
    """
    
    # Example 4: Inference
    print("\n4. Dermatology inference with TTA:")
    """
    results = dermatology_inference(
        model_path='/mnt/Test/SC202/trained_models/best_dermatology_efficientnet_b3.pth',
        image_path='/path/to/skin/lesion.jpg',
        use_tta=True
    )
    
    print(f"Top diagnosis: {results['top_diagnosis']['condition']}")
    print(f"Confidence: {results['top_diagnosis']['confidence_percentage']}")
    print(f"Interpretation: {results['clinical_interpretation']}")
    """
    
    print("\n🎯 RECOMMENDATIONS FOR DERMATOLOGY:")
    print("1. EfficientNet-B3: Best balance of accuracy and speed")
    print("2. EfficientNet-B7: Highest accuracy (if you have compute)")
    print("3. ViT-B/16: Good attention visualization for lesion analysis")
    print("4. Swin Transformer: Excellent hierarchical feature learning")
    print("5. MaxViT: Cutting-edge hybrid approach")
    
    print("\n⚡ QUICK START - Copy and run this:")
    quick_start_code = """
# Quick start with EfficientNet-B3 for dermatology
config = dermatology_config.copy()
config['model_type'] = 'efficientnet_b3'  # Proven best for medical
config['epochs'] = 50
config['img_size'] = 224
config['use_advanced_augmentation'] = True

model, history = train_dermatology_classifier(config)
"""
    print(quick_start_code)

# Run the example
if __name__ == "__main__":
    print("🔬 DERMATOLOGY STATE-OF-THE-ART MODELS")
    print("🎯 Multiple cutting-edge architectures for skin disease classification")
    print("⚡ Much better than ConvNeXt for medical imaging!")
    
    # Show example usage
    run_dermatology_sota_example()
    
    print("\n🚀 Ready to train state-of-the-art dermatology models!")
    print("💡 Try EfficientNet-B3 first - it's proven best for medical imaging!")
