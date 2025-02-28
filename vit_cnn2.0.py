import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import json
import random
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

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

class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2)  # [B, E, H*W/P^2]
        x = x.transpose(1, 2)  # [B, H*W/P^2, E]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [B, N, E]
        """
        B, N, E = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, E)  # [B, N, E]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, hidden_features=int(embed_dim * mlp_ratio), dropout=dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x

class CNNBackbone(nn.Module):
    def __init__(self, in_channels=3, dropout=0.1):
        super().__init__()
        self.features = nn.Sequential(
            CNNBlock(in_channels, 64, dropout=dropout),  # 224 -> 112
            CNNBlock(64, 128, dropout=dropout),          # 112 -> 56
            CNNBlock(128, 256, dropout=dropout),         # 56 -> 28
            CNNBlock(256, 512, dropout=dropout),         # 28 -> 14
        )
        
    def forward(self, x):
        return self.features(x)

class ViTCNNHybrid(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 num_classes=1000, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4, dropout=0.1, attn_dropout=0.0,
                 cnn_dropout=0.1):
        super().__init__()
        
        # CNN Feature Extractor - deeper network
        self.cnn_features = CNNBackbone(in_channels, dropout=cnn_dropout)
        
        # Calculate new image size after CNN feature extraction (4 max pools of 2x2)
        self.cnn_output_size = img_size // 16
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size=self.cnn_output_size,
            patch_size=patch_size // 16,  # Adjust patch size based on CNN downsampling
            in_channels=512,  # Output channels from last CNN layer
            embed_dim=embed_dim
        )
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        
        # Dropout after position embedding
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        
        # Layer Norm and MLP Head
        self.norm = nn.LayerNorm(embed_dim)
        
        # Improved classifier head with dropout
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize transformer weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        # CNN Feature Extraction
        x = self.cnn_features(x)  # [B, C', H', W']
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, N, E]
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, E]
        
        # Add position embedding and dropout
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Apply Layer Norm
        x = self.norm(x)
        
        return x[:, 0]  # Take only the cls token
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Advanced augmentation for medical images
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


# Load and preprocess data with advanced augmentation
def load_data(data_dir, img_size=224, batch_size=32, auto_augment=True, num_workers=4):
    augmentation = AdvancedAugmentation(img_size=img_size, auto_augment=auto_augment)
    
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train", 
        transform=augmentation.train_transform
    )
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val", 
        transform=augmentation.val_transform
    )
    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test", 
        transform=augmentation.val_transform
    )
    
    # Create weighted sampler for imbalanced dataset
    class_weights = compute_class_weights(train_dataset.targets)
    samples_weights = class_weights[train_dataset.targets]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
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
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    # Save class names for future reference
    with open(os.path.join(data_dir, 'classes.json'), 'w') as f:
        class_map = {i: name for i, name in enumerate(class_names)}
        json.dump(class_map, f, indent=2)
    
    return train_loader, val_loader, test_loader, num_classes, class_names


# Compute class weights for imbalanced datasets
def compute_class_weights(targets):
    class_counts = torch.bincount(torch.tensor(targets))
    total_samples = len(targets)
    class_weights = total_samples / (len(class_counts) * class_counts.float())
    return class_weights


# Mixup augmentation
def mixup_data(x, y, alpha=0.2):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training function with mixup and AMP
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=30, device='cuda', mixup_alpha=0.2, use_amp=True, 
                patience=5, model_dir='models'):
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    model = model.to(device)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    best_val_acc = 0.0
    best_epoch = -1
    
    # For early stopping
    patience_counter = 0
    
    # Track metrics
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': []
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            # Apply mixup
            if mixup_alpha > 0:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
                
            optimizer.zero_grad()
            
            # Use AMP for faster training
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    if mixup_alpha > 0:
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                if mixup_alpha > 0:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * batch_size
            
            # If using mixup, we can't count corrects directly
            if mixup_alpha > 0:
                # For simplicity, use the primary labels for accuracy
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels_a.data)
            else:
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)
        
        if scheduler is not None:
            scheduler.step()
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            val_total_samples += batch_size
            
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            
            val_running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects.double() / val_total_samples
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.item())
        
        print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
        
        # Save the best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save model
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
            
            # Save model info
            model_info = {
                'img_size': 224,
                'patch_size': 16,
                'num_classes': model.head[-1].out_features,
                'embed_dim': model.blocks[0].norm1.normalized_shape[0],
                'depth': len(model.blocks),
                'num_heads': model.blocks[0].attn.num_heads,
                'dropout': model.pos_drop.p,
                'epoch': epoch + 1,
                'best_val_acc': best_val_acc.item(),
                'training_time': time.time() - start_time
            }
            
            with open(os.path.join(model_dir, 'model_info.json'), 'w') as f:
                json.dump(model_info, f, indent=2)
                
            print(f'New best model saved with accuracy: {val_epoch_acc:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        # Also save the most recent model
        torch.save(model.state_dict(), os.path.join(model_dir, 'last_model.pth'))
    
    total_time = time.time() - start_time
    print(f'Training completed in {total_time//60:.0f}m {total_time%60:.0f}s')
    print(f'Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}')
    
    # Plot training history
    plot_training_history(history, model_dir)
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
    
    return model, history


# Plot and save training history
def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()


# Evaluate function with confusion matrix
def evaluate_model(model, test_loader, class_names, device='cuda', save_dir='models'):
    model.eval()
    all_preds = []
    all_labels = []
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = running_corrects.double() / total_samples
    print(f'Test Accuracy: {test_acc:.4f}')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'))
    
    # Also save as a readable text file
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
    
    return test_acc, cm, report


# Main training pipeline
def main(config):
    # Device configuration
    if config['use_cuda'] and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Set directories
    data_dir = config['data_dir']
    model_dir = config['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    
    # Save configuration
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data with advanced augmentation
    train_loader, val_loader, test_loader, num_classes, class_names = load_data(
        data_dir=data_dir,
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        auto_augment=config['auto_augment'],
        num_workers=config['num_workers']
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Initialize model with improved architecture
    model = ViTCNNHybrid(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        attn_dropout=config['attn_dropout'],
        cnn_dropout=config['cnn_dropout']
    )
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function - Label smoothing can help with overconfidence
    if config['label_smoothing'] > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer - AdamW with weight decay and correct scheduling
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['epochs'] // 3,
            T_mult=1,
            eta_min=config['lr'] / 100
        )
    elif config['scheduler'] == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            steps_per_epoch=len(train_loader),
            epochs=config['epochs'],
            pct_start=0.1
        )
    else:
        scheduler = None
    
    # Train model with improved training loop
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config['epochs'],
        device=device,
        mixup_alpha=config['mixup_alpha'],
        use_amp=config['use_amp'],
        patience=config['patience'],
        model_dir=model_dir
    )
    
    # Evaluate model with detailed metrics
    test_acc, cm, report = evaluate_model(
        trained_model, 
        test_loader, 
        class_names,
        device=device,
        save_dir=model_dir
    )
    
    print(f"Final test accuracy: {test_acc:.4f}")
    
    # Return results
    return {
        'model': trained_model,
        'history': history,
        'test_acc': test_acc,
        'confusion_matrix': cm,
        'report': report
    }


if __name__ == "__main__":
    # Configuration dictionary - modify these values directly
    config = {
        # Data configuration
        'data_dir': 'processed',           # Path to data with train/val/test splits
        'model_dir': 'improved_model',     # Directory to save models and results
        
        # Model parameters
        'img_size': 224,                   # Input image size
        'patch_size': 16,                  # Patch size for ViT
        'embed_dim': 768,                  # Embedding dimension (increased from 512)
        'depth': 8,                        # Number of transformer blocks (increased from 6)
        'num_heads': 12,                   # Number of attention heads (increased from 8)
        'dropout': 0.2,                    # Dropout rate (increased from 0.1)
        'attn_dropout': 0.1,               # Attention dropout rate (new parameter)
        'cnn_dropout': 0.1,                # CNN feature extractor dropout (new parameter)
        
        # Training parameters
        'batch_size': 32,                  # Batch size
        'epochs': 100,                      # Maximum number of epochs (increased from 20)
        'lr': 1e-4,                        # Learning rate (decreased from 3e-4)
        'weight_decay': 0.05,              # Weight decay for regularization
        'label_smoothing': 0.1,            # Label smoothing factor (new parameter)
        'mixup_alpha': 0.2,                # Mixup alpha parameter (new parameter)
        'auto_augment': True,              # Use AutoAugment policy (new parameter)
        'scheduler': 'cosine',             # Learning rate scheduler ('cosine' or 'onecycle')
        'patience': 10,                    # Early stopping patience (new parameter)
        
        # Performance options
        'use_cuda': True,                  # Use GPU if available
        'use_amp': True,                   # Use Automatic Mixed Precision
        'num_workers': 4                   # Number of data loading workers
    }
    
    # Print the configuration
    print("Running with configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run the training pipeline
    results = main(config)
    
    print(f"Training completed. Final test accuracy: {results['test_acc']:.4f}")
    print(f"Model and results saved to {config['model_dir']}")