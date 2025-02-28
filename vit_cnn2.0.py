import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets, models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import json
import random
import time
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Root directory - fixed path as specified
ROOT_DIR = "/mnt/Test/SC202"

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

class EnhancedCNNBackbone(nn.Module):
    """
    Enhanced CNN Backbone using a pre-trained ResNet50 model
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load a pre-trained ResNet50 model
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Use layers up to the fourth block
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # Resolution: 56x56, Channels: 256
            resnet.layer2,  # Resolution: 28x28, Channels: 512
            resnet.layer3,  # Resolution: 14x14, Channels: 1024
            resnet.layer4   # Resolution: 7x7, Channels: 2048
        )
        
        # Freeze early layers to preserve learned features
        for param in self.features[:6].parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.features(x)

class ImprovedPatchEmbedding(nn.Module):
    """
    Improved patch embedding with layer normalization
    """
    def __init__(self, in_channels=2048, patch_size=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, E, H/P, W/P]
        x = x.flatten(2)  # [B, E, H*W/P^2]
        x = x.transpose(1, 2)  # [B, H*W/P^2, E]
        x = self.norm(x)
        return x

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with improved implementation
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Use a single projection for efficiency
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [B, N, E]
        """
        B, N, E = x.shape
        
        # Calculate QKV projections in one go for efficiency
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values and reshape
        x = (attn @ v).transpose(1, 2).reshape(B, N, E)  # [B, N, E]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class FeedForward(nn.Module):
    """
    Improved MLP block with GELU activation and layer norm
    """
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Enhanced transformer block with pre-norm architecture
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4, dropout=0.1, attn_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(
            dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            dropout=dropout
        )
        
    def forward(self, x):
        # Pre-norm architecture (better training stability)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ImprovedViTCNNHybrid(nn.Module):
    """
    Improved ViT-CNN Hybrid with pre-trained CNN backbone
    """
    def __init__(self, img_size=224, patch_size=1, in_channels=3, 
                 num_classes=8, embed_dim=768, depth=12, 
                 num_heads=12, mlp_ratio=4, dropout=0.1, attn_dropout=0.1,
                 cnn_pretrained=True):
        super().__init__()
        
        # Enhanced CNN Feature Extractor - pre-trained
        self.cnn_features = EnhancedCNNBackbone(pretrained=cnn_pretrained)
        self.cnn_output_size = img_size // 32  # ResNet50 downsamples by factor of 32
        
        # Improved Patch Embedding
        self.patch_embed = ImprovedPatchEmbedding(
            in_channels=2048,  # ResNet50 output channels
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        
        # Class token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (self.cnn_output_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer Encoder with better initialization
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, attn_dropout)
            for _ in range(depth)
        ])
        
        # Improved classifier head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
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
    
    def forward_features(self, x):
        # CNN Feature Extraction
        x = self.cnn_features(x)  # [B, 2048, H/32, W/32]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, (H/32)*(W/32), E]
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+(H/32)*(W/32), E]
        
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


# Enhanced configuration optimized for skin disease classification
def get_optimal_config():
    # Define model directory within ROOT_DIR
    model_dir = os.path.join(ROOT_DIR, 'improved_vitcnn_model')
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    return {
        # Data configuration
        'train_data_path': "/mnt/Test/SC202/SkinDisease/SkinDisease/train",
        'test_data_path': "/mnt/Test/SC202/SkinDisease/SkinDisease/test",
        'model_dir': model_dir,
        'validation_split': 0.15,  # Increased validation split for better evaluation
        
        # Model parameters - optimized for skin disease classification
        'img_size': 224,
        'patch_size': 1,  # Use 1x1 patches for fine-grained features
        'embed_dim': 512,  # Reduced embedding dimension for better generalization
        'depth': 6,       # Shallower transformer (sufficient for this task)
        'num_heads': 8,   # Fewer attention heads
        'dropout': 0.2,   # Increased dropout for better regularization
        'attn_dropout': 0.1,
        'cnn_pretrained': True,  # Use pre-trained backbone
        
        # Training parameters - optimal for skin dataset
        'batch_size': 8,   # Smaller batch size to prevent OOM errors
        'epochs': 50,      # More epochs for thorough training
        'lr': 5e-5,        # Lower learning rate for stable training
        'weight_decay': 0.01,
        'label_smoothing': 0.1,
        'mixup_alpha': 0.4,  # Stronger mixup for better generalization
        'auto_augment': True,
        'scheduler': 'cosine',
        'patience': 15,    # More patience to avoid early stopping
        
        # Performance options
        'use_cuda': True,
        'use_amp': True,
        'num_workers': 2   # Reduced workers to prevent memory issues
    }


# Modified loading function to handle memory constraints
def load_data_memory_efficient(train_data_path, test_data_path, img_size=224, batch_size=8, 
                  auto_augment=True, num_workers=2, validation_split=0.15):
    """
    Memory efficient data loading function
    """
    # Advanced augmentation with memory efficiency in mind
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Direct resize instead of larger crop
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3)
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset_full = datasets.ImageFolder(
        root=train_data_path,
        transform=train_transforms
    )
    
    # Get class names
    class_names = train_dataset_full.classes
    num_classes = len(class_names)
    
    # Calculate train and validation sizes
    train_size = int((1 - validation_split) * len(train_dataset_full))
    val_size = len(train_dataset_full) - train_size
    
    # Use random_split to get indices
    train_indices, val_indices = random_split(
        range(len(train_dataset_full)),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create datasets
    train_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(
            root=train_data_path,
            transform=train_transforms
        ),
        train_indices.indices
    )
    
    val_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(
            root=train_data_path,
            transform=test_transforms
        ),
        val_indices.indices
    )
    
    # Load test dataset
    test_dataset = datasets.ImageFolder(
        root=test_data_path,
        transform=test_transforms
    )
    
    # Create memory-efficient data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Use shuffle instead of sampler to save memory
        num_workers=num_workers,
        pin_memory=False,  # Disable pin_memory to save memory
        drop_last=True     # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader, test_loader, num_classes, class_names


# Mixup augmentation function
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


# Memory-efficient training loop
def train_model_efficient(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                          num_epochs=30, device='cuda', mixup_alpha=0.2, use_amp=True,
                          patience=15, model_dir=None):
    if model_dir is None:
        model_dir = os.path.join(ROOT_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Models will be saved to: {model_dir}")
    
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
        'val_acc': [],
        'lr': []  # Track learning rate changes
    }
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        # Use tqdm for progress tracking
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), 
                            desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        # Process batches
        for i, (inputs, labels) in train_progress:
            try:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                # Apply mixup
                if mixup_alpha > 0:
                    # Mixup implementation
                    mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, mixup_alpha)
                    
                    optimizer.zero_grad()
                    
                    # Use AMP for faster training
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(mixed_inputs)
                            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(mixed_inputs)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                        loss.backward()
                        optimizer.step()
                    
                    # Calculate accuracy for mixup (approximation)
                    _, preds = torch.max(outputs, 1)
                    running_corrects += (lam * torch.sum(preds == labels_a) + 
                                       (1 - lam) * torch.sum(preds == labels_b)).item()
                else:
                    optimizer.zero_grad()
                    
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                    
                    # Calculate accuracy 
                    _, preds = torch.max(outputs, 1)
                    running_corrects += torch.sum(preds == labels).item()
                
                running_loss += loss.item() * batch_size
                
                # Update progress
                train_progress.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })
                
                # Release memory
                del inputs, outputs
                if mixup_alpha > 0:
                    del mixed_inputs, labels_a, labels_b
                torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"WARNING: GPU out of memory in batch {i}, skipping...")
                    # Clear cache and skip this batch
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            history['lr'].append(current_lr)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects / total_samples
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        val_progress = tqdm(enumerate(val_loader), total=len(val_loader), 
                          desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        for i, (inputs, labels) in val_progress:
            try:
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
                val_running_corrects += torch.sum(preds == labels).item()
                
                # Update progress
                val_progress.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })
                
                # Release memory
                del inputs, outputs
                torch.cuda.empty_cache()
            
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"WARNING: GPU out of memory in validation batch {i}, skipping...")
                    # Clear cache and skip this batch
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects / val_total_samples
        
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        
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
                'epoch': epoch + 1,
                'val_acc': val_epoch_acc,
                'val_loss': val_epoch_loss,
                'train_acc': epoch_acc,
                'train_loss': epoch_loss,
                'best_val_acc': best_val_acc,
                'img_size': 224,
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
        
        # Plot and save training history after each epoch
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            plot_training_history(history, model_dir)
        
        # Clear memory at the end of each epoch
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    print(f'Training completed in {total_time//60:.0f}m {total_time%60:.0f}s')
    print(f'Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}')
    
    # Load the best model for final evaluation
    model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth')))
    
    return model, history


# Plot and save training history
def plot_training_history(history, save_dir):
    plt.figure(figsize=(15, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot learning rate if available
    if 'lr' in history and len(history['lr']) > 0:
        plt.subplot(1, 3, 3)
        plt.plot(history['lr'], label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300)
    plt.close()


# Memory-efficient evaluation function
def evaluate_model_memory_efficient(model, test_loader, class_names, device='cuda', save_dir=None):
    """
    Memory-efficient model evaluation function
    """
    if save_dir is None:
        save_dir = os.path.join(ROOT_DIR, 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    all_preds = []
    all_labels = []
    running_corrects = 0
    total_samples = 0
    
    print("Evaluating model...")
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(test_loader, desc="Evaluating")):
            try:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)
                total_samples += batch_size
                
                # Use AMP for memory efficiency
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                
                running_corrects += torch.sum(preds == labels).item()
                
                # Add to lists
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Release memory
                del inputs, outputs, preds
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(f"WARNING: GPU out of memory in evaluation batch {i}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    test_acc = running_corrects / total_samples
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
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Generate normalized confusion matrix (in percentage)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.round(cm_norm * 100, 1)  # Convert to percentages
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_norm, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_normalized.png'), dpi=300)
    plt.close()
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    
    # Create a more detailed report dataframe
    report_df = pd.DataFrame(report).transpose()
    
    # Save reports
    report_df.to_csv(os.path.join(save_dir, 'classification_report.csv'))
    
    # Also save as readable text file
    with open(os.path.join(save_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Create per-class accuracy bar chart
    plt.figure(figsize=(14, 8))
    class_accuracies = [report[name]['precision'] for name in class_names]
    bars = plt.bar(class_names, class_accuracies, color='skyblue')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.2f}',
            ha='center',
            fontsize=9
        )
    
    plt.title('Per-Class Accuracy', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=300)
    plt.close()
    
    # Save a summary of the evaluation
    with open(os.path.join(save_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write(f"Overall Test Accuracy: {test_acc:.4f}\n\n")
        f.write("Per-class Performance:\n")
        for i, class_name in enumerate(class_names):
            class_stats = report[class_name]
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {class_stats['precision']:.4f}\n")
            f.write(f"  Recall: {class_stats['recall']:.4f}\n")
            f.write(f"  F1-Score: {class_stats['f1-score']:.4f}\n")
            f.write(f"  Samples: {class_stats['support']}\n\n")
    
    return test_acc, cm, report


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get optimal configuration
    config = get_optimal_config()
    
    # Make sure all directories exist
    os.makedirs(config['model_dir'], exist_ok=True)
    
    # Save configuration to model directory
    with open(os.path.join(config['model_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Log beginning of training process
    print(f"Starting training process for skin disease classification")
    print(f"Using ROOT_DIR: {ROOT_DIR}")
    print(f"Models will be saved to: {config['model_dir']}")
    
    # Load data with memory-efficient function
    train_loader, val_loader, test_loader, num_classes, class_names = load_data_memory_efficient(
        train_data_path=config['train_data_path'],
        test_data_path=config['test_data_path'],
        img_size=config['img_size'],
        batch_size=config['batch_size'],
        auto_augment=config['auto_augment'],
        num_workers=config['num_workers'],
        validation_split=config['validation_split']
    )
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}")
    
    # Save class names for future reference
    with open(os.path.join(config['model_dir'], 'classes.json'), 'w') as f:
        class_map = {i: name for i, name in enumerate(class_names)}
        json.dump(class_map, f, indent=2)
    
    # Initialize model
    model = ImprovedViTCNNHybrid(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        in_channels=3,
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        dropout=config['dropout'],
        attn_dropout=config['attn_dropout'],
        cnn_pretrained=config['cnn_pretrained']
    )
    
    # Print model summary
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")
    
    # Loss function with label smoothing
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    
    # Optimizer with weight decay
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
    
    # Train model with memory-efficient training
    trained_model, history = train_model_efficient(
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
        model_dir=config['model_dir']
    )
    
    # Evaluate model
    test_acc, cm, report = evaluate_model_memory_efficient(
        trained_model, 
        test_loader, 
        class_names,
        device=device,
        save_dir=config['model_dir']
    )
    
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"Model and results saved to: {config['model_dir']}")
    
    # Return results
    return {
        'model': trained_model,
        'history': history,
        'test_acc': test_acc,
        'confusion_matrix': cm,
        'report': report
    }


if __name__ == "__main__":
    # Make sure ROOT_DIR exists
    os.makedirs(ROOT_DIR, exist_ok=True)
    
    # Run the main function
    results = main()
    
    print("Training and evaluation completed successfully!")
    print(f"Best test accuracy: {results['test_acc']:.4f}")
