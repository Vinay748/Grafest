import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Style setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
print("\n All libraries imported successfully!")

# Dataset path configuration
DATASET_ROOT = r"d:\NEURAL NEXUS\garbage_dataset\arrwanged"

# Category mappings
CATEGORY_MAP = {
    '1': 'Category 1 - Highly Polluted',
    '2': 'Category 2 - Polluted',
    '3': 'Category 3 - Moderately Clean',
    '4': 'Category 4 - Clean',
    '5': 'Category 5 - Very Clean'
}

POLLUTION_LEVEL_MAP = {
    '1': ('Critical', 9.5, 'High'),
    '2': ('Severe', 7.5, 'High'),
    '3': ('Moderate', 5.0, 'Medium'),
    '4': ('Low', 2.5, 'Low'),
    '5': ('Minimal', 0.5, 'Low')
}

# Waste type estimation profiles per category (simulated based on pollution level)
WASTE_TYPE_PROFILES = {
    '1': {'Plastic': 0.35, 'Organic': 0.25, 'Metallic': 0.10, 'E-waste': 0.08, 'Hazardous': 0.12, 'Anomalies': 0.10},
    '2': {'Plastic': 0.30, 'Organic': 0.30, 'Metallic': 0.12, 'E-waste': 0.06, 'Hazardous': 0.07, 'Anomalies': 0.15},
    '3': {'Plastic': 0.25, 'Organic': 0.35, 'Metallic': 0.08, 'E-waste': 0.04, 'Hazardous': 0.03, 'Anomalies': 0.25},
    '4': {'Plastic': 0.15, 'Organic': 0.40, 'Metallic': 0.05, 'E-waste': 0.02, 'Hazardous': 0.01, 'Anomalies': 0.37},
    '5': {'Plastic': 0.05, 'Organic': 0.10, 'Metallic': 0.02, 'E-waste': 0.01, 'Hazardous': 0.00, 'Anomalies': 0.82}
}

# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
SKIPPED_EXTENSIONS = {'.heic', '.avif'}  # Will be filtered out

# Collect all image paths
all_images = []
all_labels = []
skipped_count = 0

for category in sorted(os.listdir(DATASET_ROOT)):
    category_path = os.path.join(DATASET_ROOT, category)
    if not os.path.isdir(category_path):
        continue
    for img_file in os.listdir(category_path):
        ext = os.path.splitext(img_file)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            all_images.append(os.path.join(category_path, img_file))
            all_labels.append(int(category) - 1)  # 0-indexed
        elif ext in SKIPPED_EXTENSIONS:
            skipped_count += 1

print(f" Dataset Statistics:")
print(f"   Total usable images: {len(all_images)}")
print(f"   Skipped (unsupported HEIC/AVIF): {skipped_count}")
print(f"\n Class Distribution:")

label_counts = Counter(all_labels)
for label in sorted(label_counts.keys()):
    cat_name = CATEGORY_MAP[str(label + 1)]
    count = label_counts[label]
    bar = '█' * (count // 10)
    print(f"   {cat_name}: {count} images {bar}")

fig = plt.figure(figsize=(20, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# --- Plot 1: Class Distribution Bar Chart ---
ax1 = fig.add_subplot(gs[0, 0])
categories = [f'Cat {i+1}' for i in sorted(label_counts.keys())]
counts = [label_counts[i] for i in sorted(label_counts.keys())]
colors = ['#FF4444', '#FF8800', '#FFCC00', '#44BB44', '#2288FF']
bars = ax1.bar(categories, counts, color=colors, edgecolor='white', linewidth=1.5)
ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Images')
for bar, count in zip(bars, counts):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
             str(count), ha='center', va='bottom', fontweight='bold')

# --- Plot 2: Pollution Level Pie Chart ---
ax2 = fig.add_subplot(gs[0, 1])
pollution_counts = {'High': 0, 'Medium': 0, 'Low': 0}
for label in all_labels:
    _, _, level = POLLUTION_LEVEL_MAP[str(label + 1)]
    pollution_counts[level] += 1
p_labels = list(pollution_counts.keys())
p_values = list(pollution_counts.values())
p_colors = ['#FF4444', '#FFAA00', '#44BB44']
wedges, texts, autotexts = ax2.pie(p_values, labels=p_labels, colors=p_colors, 
                                    autopct='%1.1f%%', startangle=90,
                                    pctdistance=0.85, textprops={'fontsize': 11})
centre_circle = plt.Circle((0, 0), 0.55, fc='white')
ax2.add_artist(centre_circle)
ax2.set_title('Pollution Level Distribution', fontsize=14, fontweight='bold')

# --- Plot 3: Waste Type Composition (Average) ---
ax3 = fig.add_subplot(gs[0, 2])
avg_waste = {wtype: 0 for wtype in WASTE_TYPE_PROFILES['1'].keys()}
for label in all_labels:
    profile = WASTE_TYPE_PROFILES[str(label + 1)]
    for wtype, pct in profile.items():
        avg_waste[wtype] += pct
total = len(all_labels)
avg_waste = {k: v/total*100 for k, v in avg_waste.items()}
w_colors = ['#3498db', '#2ecc71', '#95a5a6', '#e67e22', '#e74c3c', '#9b59b6']
bars3 = ax3.barh(list(avg_waste.keys()), list(avg_waste.values()), color=w_colors)
ax3.set_title('Avg Waste Composition (%)', fontsize=14, fontweight='bold')
ax3.set_xlabel('Percentage')
for bar, val in zip(bars3, avg_waste.values()):
    ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2.,
             f'{val:.1f}%', ha='left', va='center', fontsize=10)

# --- Plot 4-6: Sample Images per Category ---
for idx, cat in enumerate(['1', '3', '5']):
    ax = fig.add_subplot(gs[1, idx])
    cat_images = [all_images[i] for i in range(len(all_images)) if all_labels[i] == int(cat) - 1]
    if cat_images:
        sample_path = random.choice(cat_images)
        try:
            img = Image.open(sample_path).convert('RGB')
            img = img.resize((300, 300))
            ax.imshow(img)
            ax.set_title(f'{CATEGORY_MAP[cat]}', fontsize=11, fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\n{str(e)[:30]}', 
                    ha='center', va='center', transform=ax.transAxes)
    ax.axis('off')

plt.suptitle(' Neural Nexus — Dataset Analysis Dashboard', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'd:\NEURAL NEXUS\dataset_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n Dataset analysis complete! Plot saved.")

class WasteDataset(Dataset):
    """Custom PyTorch dataset with robust image loading and augmentation."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # Fallback: create a blank image if file is corrupted
            image = Image.new('RGB', (224, 224), (128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ========== Data Augmentation Strategy ==========
# Training: Heavy augmentation for robustness
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

# Validation/Test: No augmentation, only resize & normalize
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print(" Dataset class and augmentation pipeline defined!")
print(f"\n Training Augmentations:")
print("   • Random Resized Crop (224x224)")
print("   • Random Horizontal/Vertical Flip")
print("   • Random Rotation (±20°)")
print("   • Color Jitter (brightness, contrast, saturation, hue)")
print("   • Random Affine Translation")
print("   • Random Grayscale")
print("   • Gaussian Blur")
print("   • Random Erasing (cutout)")
print("   • ImageNet Normalization")

# Stratified split: 70% train, 15% validation, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    all_images, all_labels, test_size=0.15, stratify=all_labels, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=SEED  # ~15% of total
)

print(f" Data Split:")
print(f"   Training:   {len(X_train)} images ({len(X_train)/len(all_images)*100:.1f}%)")
print(f"   Validation: {len(X_val)} images ({len(X_val)/len(all_images)*100:.1f}%)")
print(f"   Test:       {len(X_test)} images ({len(X_test)/len(all_images)*100:.1f}%)")

# Create datasets
train_dataset = WasteDataset(X_train, y_train, transform=train_transform)
val_dataset = WasteDataset(X_val, y_val, transform=val_transform)
test_dataset = WasteDataset(X_test, y_test, transform=val_transform)

# Handle class imbalance with WeightedRandomSampler
train_label_counts = Counter(y_train)
class_weights = {cls: 1.0 / count for cls, count in train_label_counts.items()}
sample_weights = [class_weights[label] for label in y_train]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# DataLoaders
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, 
                          num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                         num_workers=0, pin_memory=True)

print(f"\n DataLoaders created:")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Training batches: {len(train_loader)}")
print(f"   Validation batches: {len(val_loader)}")
print(f"   Test batches: {len(test_loader)}")
print(f"   Class balancing: WeightedRandomSampler ")

class WasteIntelligenceModel(nn.Module):
    """
    EfficientNet-B0 based model with custom multi-task head.
    
    Outputs:
    - Cleanliness classification (5 categories)
    - Feature embeddings for waste type analysis
    """
    
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super(WasteIntelligenceModel, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get the feature dimension from the classifier
        in_features = self.backbone.classifier[1].in_features
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )
        
        # Feature extractor for waste type analysis
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        embeddings = self.feature_extractor(features)
        return logits, embeddings
    
    def get_attention_features(self, x):
        """Extract features from the last conv layer for Grad-CAM."""
        return self.backbone.features(x)

# Initialize model
model = WasteIntelligenceModel(num_classes=5).to(device)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f" Model Architecture: EfficientNet-B0 + Custom Head")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Backbone: EfficientNet-B0 (ImageNet pretrained)")
print(f"   Classifier: 1280 → 512 → 256 → 5")
print(f"   Feature Extractor: 1280 → 256 → 128 (embeddings)")
print(f"   Dropout: Multi-level (0.4, 0.2, 0.1)")
print(f"   Batch Normalization: ")

# Label smoothing cross entropy for better generalization
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # One-hot encoding with label smoothing
        one_hot = torch.zeros_like(log_preds).scatter_(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        loss = -(one_hot * log_preds).sum(dim=-1)
        
        if self.weight is not None:
            weight = self.weight[target]
            loss = loss * weight
        
        return loss.mean()

# Class weights for cost-sensitive learning
class_counts = torch.tensor([label_counts[i] for i in range(5)], dtype=torch.float32)
class_weights_tensor = (1.0 / class_counts) * class_counts.sum() / len(class_counts)
class_weights_tensor = class_weights_tensor.to(device)

# Loss function
criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights_tensor)

# Optimizer: AdamW with weight decay
optimizer = optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-4},      # Backbone (lower LR)
    {'params': model.classifier.parameters(), 'lr': 5e-4},     # Classifier (higher LR)
    {'params': model.feature_extractor.parameters(), 'lr': 5e-4}
], weight_decay=1e-4)

# Learning rate scheduler: Cosine annealing with warm restarts
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

print(" Training Configuration:")
print(f"   Loss: Label Smoothing Cross Entropy (ε=0.1)")
print(f"   Optimizer: AdamW (weight_decay=1e-4)")
print(f"   LR - Backbone: 1e-4")
print(f"   LR - Classifier: 5e-4")
print(f"   Scheduler: CosineAnnealingWarmRestarts (T0=5)")
print(f"   Class weights: {class_weights_tensor.cpu().numpy().round(3)}")

# Training hyperparameters
NUM_EPOCHS = 25
PATIENCE = 7  # Early stopping patience
MIN_DELTA = 0.001  # Minimum improvement

# Training history
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [], 'val_acc': [],
    'train_f1': [], 'val_f1': [],
    'lr': []
}

best_val_acc = 0.0
best_val_f1 = 0.0
patience_counter = 0
best_model_state = None

print(f" Starting Training: {NUM_EPOCHS} epochs")
print("=" * 80)

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    
    # ========== Training Phase ==========
    model.train()
    train_loss = 0.0
    train_preds, train_targets = [], []
    
    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]', 
                     leave=False, ncols=100)
    
    for images, labels in train_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())
        
        train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # ========== Validation Phase ==========
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []
    val_probs = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits, _ = model(images)
            loss = criterion(logits, labels)
            
            val_loss += loss.item() * images.size(0)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    train_loss /= len(train_dataset)
    val_loss /= len(val_dataset)
    train_acc = accuracy_score(train_targets, train_preds)
    val_acc = accuracy_score(val_targets, val_preds)
    train_f1 = f1_score(train_targets, train_preds, average='weighted')
    val_f1 = f1_score(val_targets, val_preds, average='weighted')
    
    # Learning rate
    current_lr = optimizer.param_groups[0]['lr']
    scheduler.step()
    
    # Save history
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_f1'].append(train_f1)
    history['val_f1'].append(val_f1)
    history['lr'].append(current_lr)
    
    epoch_time = time.time() - epoch_start
    
    # Print epoch summary
    improved = ''
    if val_acc > best_val_acc + MIN_DELTA:
        best_val_acc = val_acc
        best_val_f1 = val_f1
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        improved = ' BEST'
        torch.save(model.state_dict(), r'd:\NEURAL NEXUS\best_model.pth')
    else:
        patience_counter += 1
    
    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS} │ "
          f"Train Loss: {train_loss:.4f} │ Val Loss: {val_loss:.4f} │ "
          f"Train Acc: {train_acc:.4f} │ Val Acc: {val_acc:.4f} │ "
          f"Val F1: {val_f1:.4f} │ LR: {current_lr:.6f} │ "
          f"Time: {epoch_time:.1f}s{improved}")
    
    # Early stopping
    if patience_counter >= PATIENCE:
        print(f"\n  Early stopping triggered at epoch {epoch+1} (patience={PATIENCE})")
        break

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

print(f"\n{'='*80}")
print(f" Training Complete!")
print(f"   Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"   Best Validation F1 Score: {best_val_f1:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Loss curves
ax1 = axes[0, 0]
ax1.plot(history['train_loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.8)
ax1.plot(history['val_loss'], 'r-', linewidth=2, label='Val Loss', alpha=0.8)
ax1.fill_between(range(len(history['train_loss'])), history['train_loss'], alpha=0.1, color='blue')
ax1.fill_between(range(len(history['val_loss'])), history['val_loss'], alpha=0.1, color='red')
ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)

# Accuracy curves
ax2 = axes[0, 1]
ax2.plot(history['train_acc'], 'b-', linewidth=2, label='Train Acc', alpha=0.8)
ax2.plot(history['val_acc'], 'r-', linewidth=2, label='Val Acc', alpha=0.8)
ax2.axhline(y=best_val_acc, color='green', linestyle='--', alpha=0.5, label=f'Best Val Acc: {best_val_acc:.4f}')
ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

# F1 Score curves
ax3 = axes[1, 0]
ax3.plot(history['train_f1'], 'b-', linewidth=2, label='Train F1', alpha=0.8)
ax3.plot(history['val_f1'], 'r-', linewidth=2, label='Val F1', alpha=0.8)
ax3.set_title('F1 Score Curves', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('F1 Score')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)

# Learning Rate Schedule
ax4 = axes[1, 1]
ax4.plot(history['lr'], 'g-', linewidth=2, alpha=0.8)
ax4.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Learning Rate')
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3)

plt.suptitle(' Neural Nexus — Training Dashboard', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'd:\NEURAL NEXUS\training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

model.eval()

test_preds = []
test_targets = []
test_probs = []
test_embeddings = []

print(" Evaluating model on test set...")

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc='Testing', ncols=80):
        images = images.to(device)
        labels = labels.to(device)
        
        logits, embeddings = model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.cpu().numpy())
        test_probs.extend(probs.cpu().numpy())
        test_embeddings.extend(embeddings.cpu().numpy())

test_probs = np.array(test_probs)
test_preds = np.array(test_preds)
test_targets = np.array(test_targets)
test_embeddings = np.array(test_embeddings)

# ========== Core Metrics ==========
test_accuracy = accuracy_score(test_targets, test_preds)
test_f1 = f1_score(test_targets, test_preds, average='weighted')
test_precision = precision_score(test_targets, test_preds, average='weighted')
test_recall = recall_score(test_targets, test_preds, average='weighted')

# Confidence metrics
max_probs = np.max(test_probs, axis=1)
mean_confidence = np.mean(max_probs)
median_confidence = np.median(max_probs)
correct_mask = test_preds == test_targets
correct_confidence = np.mean(max_probs[correct_mask]) if correct_mask.sum() > 0 else 0
incorrect_confidence = np.mean(max_probs[~correct_mask]) if (~correct_mask).sum() > 0 else 0

print("\n" + "=" * 80)
print(" MODEL PERFORMANCE REPORT")
print("=" * 80)
print(f"\n Overall Metrics:")
print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"   F1 Score:  {test_f1:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall:    {test_recall:.4f}")

print(f"\n Confidence Metrics:")
print(f"   Mean Confidence:     {mean_confidence:.4f} ({mean_confidence*100:.2f}%)")
print(f"   Median Confidence:   {median_confidence:.4f} ({median_confidence*100:.2f}%)")
print(f"   Correct Predictions: {correct_confidence:.4f} ({correct_confidence*100:.2f}%)")
print(f"   Wrong Predictions:   {incorrect_confidence:.4f} ({incorrect_confidence*100:.2f}%)")

print(f"\n Per-Class Classification Report:")
class_names = [f'Cat {i+1}' for i in range(5)]
print(classification_report(test_targets, test_preds, target_names=class_names, digits=4))

fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# --- Confusion Matrix ---
ax1 = axes[0]
cm = confusion_matrix(test_targets, test_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='YlOrRd',
            xticklabels=class_names, yticklabels=class_names,
            ax=ax1, cbar_kws={'label': 'Percentage'},
            linewidths=0.5, linecolor='white')
ax1.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('Actual', fontsize=12)

# --- Confidence Distribution ---
ax2 = axes[1]
ax2.hist(max_probs[correct_mask], bins=30, alpha=0.7, color='#2ecc71', 
         label=f'Correct ({correct_mask.sum()})', density=True, edgecolor='white')
ax2.hist(max_probs[~correct_mask], bins=30, alpha=0.7, color='#e74c3c',
         label=f'Incorrect ({(~correct_mask).sum()})', density=True, edgecolor='white')
ax2.axvline(x=mean_confidence, color='blue', linestyle='--', linewidth=2,
            label=f'Mean: {mean_confidence:.3f}')
ax2.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Confidence', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.legend(fontsize=10)

# --- Per-Class Accuracy ---
ax3 = axes[2]
per_class_acc = []
per_class_conf = []
for i in range(5):
    mask = test_targets == i
    if mask.sum() > 0:
        pc_acc = (test_preds[mask] == i).mean()
        pc_conf = max_probs[mask].mean()
    else:
        pc_acc = 0
        pc_conf = 0
    per_class_acc.append(pc_acc)
    per_class_conf.append(pc_conf)

x_pos = np.arange(5)
width = 0.35
bars1 = ax3.bar(x_pos - width/2, per_class_acc, width, label='Accuracy', 
                color='#3498db', edgecolor='white')
bars2 = ax3.bar(x_pos + width/2, per_class_conf, width, label='Avg Confidence',
                color='#e67e22', edgecolor='white')
ax3.set_title('Per-Class Accuracy & Confidence', fontsize=14, fontweight='bold')
ax3.set_xlabel('Category', fontsize=12)
ax3.set_ylabel('Score', fontsize=12)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(class_names)
ax3.legend(fontsize=11)
ax3.set_ylim(0, 1.15)

for bar in bars1:
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

plt.suptitle(' Neural Nexus — Evaluation Dashboard', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'd:\NEURAL NEXUS\evaluation_results.png', dpi=150, bbox_inches='tight')
plt.show()

class WasteIntelligenceEngine:
    """
    Comprehensive waste analysis engine that provides:
    - Waste type segregation (Plastic, Metallic, E-waste, Hazardous, Organic, Anomalies)
    - Pollution severity scoring (1-10)
    - Pollution impact assessment
    - Priority-based action recommendations
    """
    
    WASTE_TYPES = ['Plastic', 'Organic', 'Metallic', 'E-waste', 'Hazardous', 'Anomalies']
    
    WASTE_PROFILES = {
        0: {'Plastic': 0.35, 'Organic': 0.25, 'Metallic': 0.10, 'E-waste': 0.08, 'Hazardous': 0.12, 'Anomalies': 0.10},
        1: {'Plastic': 0.30, 'Organic': 0.30, 'Metallic': 0.12, 'E-waste': 0.06, 'Hazardous': 0.07, 'Anomalies': 0.15},
        2: {'Plastic': 0.25, 'Organic': 0.35, 'Metallic': 0.08, 'E-waste': 0.04, 'Hazardous': 0.03, 'Anomalies': 0.25},
        3: {'Plastic': 0.15, 'Organic': 0.40, 'Metallic': 0.05, 'E-waste': 0.02, 'Hazardous': 0.01, 'Anomalies': 0.37},
        4: {'Plastic': 0.05, 'Organic': 0.10, 'Metallic': 0.02, 'E-waste': 0.01, 'Hazardous': 0.00, 'Anomalies': 0.82}
    }
    
    POLLUTION_SCORES = {
        0: (9.0, 10.0, 'Critical', 'High'),
        1: (7.0, 8.5, 'Severe', 'High'),
        2: (4.5, 6.5, 'Moderate', 'Medium'),
        3: (2.0, 4.0, 'Low', 'Low'),
        4: (0.0, 1.5, 'Minimal', 'Low')
    }
    
    IMPACT_ASSESSMENT = {
        0: {
            'environmental': 'CRITICAL — Severe soil/water contamination, toxic leachate risk',
            'health': 'HIGH RISK — Disease vectors, respiratory hazards, chemical exposure',
            'ecological': 'DEVASTATING — Wildlife habitat destruction, biodiversity loss',
            'economic': 'HIGH COST — Extensive remediation needed ($$$)',
        },
        1: {
            'environmental': 'SEVERE — Significant contamination, groundwater risk',
            'health': 'MODERATE-HIGH — Pest breeding, odor issues, infection risk',
            'ecological': 'SERIOUS — Local ecosystem disruption',
            'economic': 'MODERATE-HIGH — Professional cleanup required ($$)',
        },
        2: {
            'environmental': 'MODERATE — Localized pollution, manageable with intervention',
            'health': 'MODERATE — Minor health concerns, aesthetic degradation',
            'ecological': 'MODERATE — Some habitat disturbance',
            'economic': 'MODERATE — Targeted cleanup operations ($)',
        },
        3: {
            'environmental': 'LOW — Minor scattered waste, minimal contamination',
            'health': 'LOW — Minimal health impact',
            'ecological': 'LOW — Minimal ecological impact',
            'economic': 'LOW — Routine maintenance sufficient',
        },
        4: {
            'environmental': 'MINIMAL — Clean environment, well-maintained',
            'health': 'NEGLIGIBLE — Safe, hygienic conditions',
            'ecological': 'HEALTHY — Natural ecosystem intact',
            'economic': 'MINIMAL — Standard maintenance only',
        }
    }
    
    ACTION_RECOMMENDATIONS = {
        0: [
            ' EMERGENCY: Deploy hazardous waste cleanup team IMMEDIATELY',
            ' Isolate area — potential toxic/chemical contamination',
            ' Conduct soil and water contamination testing',
            ' Deploy heavy machinery for waste removal',
            ' Report to environmental regulatory authorities',
            ' Segregate hazardous materials for specialized disposal',
            ' Plan long-term soil remediation program',
            ' Install monitoring sensors for real-time tracking',
        ],
        1: [
            ' HIGH PRIORITY: Schedule professional cleanup within 48 hours',
            ' Deploy waste segregation teams (plastic, metal, organic)',
            ' Arrange bulk waste transport vehicles',
            ' Set up on-site recycling sorting station',
            ' Pest control treatment in affected area',
            ' Alert local community via notification system',
            ' Document for environmental compliance records',
        ],
        2: [
            ' MODERATE: Schedule cleanup within 1 week',
            ' Increase waste bin capacity in the area',
            ' Promote recycling awareness campaigns',
            ' Install surveillance for illegal dumping prevention',
            ' Regular sweeping and maintenance schedule',
            ' Track waste generation patterns',
        ],
        3: [
            ' LOW PRIORITY: Routine maintenance adequate',
            ' Ensure adequate waste bin availability',
            ' Continue regular cleaning schedule',
            ' Maintain recycling infrastructure',
            ' Monitor for any degradation trends',
        ],
        4: [
            ' EXCELLENT: Area is well-maintained!',
            ' Continue current maintenance practices',
            ' Use as benchmark/reference area',
            ' Recognize and reward maintenance teams',
            ' Replicate best practices to other areas',
        ]
    }
    
    @staticmethod
    def analyze(predicted_class, confidence, class_probs):
        """Generate comprehensive waste analysis report."""
        
        # Waste type composition (weighted by class probabilities)
        waste_composition = {wtype: 0.0 for wtype in WasteIntelligenceEngine.WASTE_TYPES}
        for cls_idx in range(5):
            cls_prob = class_probs[cls_idx]
            profile = WasteIntelligenceEngine.WASTE_PROFILES[cls_idx]
            for wtype, pct in profile.items():
                waste_composition[wtype] += pct * cls_prob
        
        # Normalize
        total = sum(waste_composition.values())
        waste_composition = {k: v/total for k, v in waste_composition.items()}
        
        # Pollution score (weighted by probabilities)
        pollution_score = 0.0
        for cls_idx in range(5):
            low, high, _, _ = WasteIntelligenceEngine.POLLUTION_SCORES[cls_idx]
            mid = (low + high) / 2
            pollution_score += mid * class_probs[cls_idx]
        
        _, _, severity, level = WasteIntelligenceEngine.POLLUTION_SCORES[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'category_name': CATEGORY_MAP[str(predicted_class + 1)],
            'confidence': confidence,
            'waste_composition': waste_composition,
            'pollution_score': round(pollution_score, 1),
            'pollution_severity': severity,
            'pollution_level': level,
            'impact': WasteIntelligenceEngine.IMPACT_ASSESSMENT[predicted_class],
            'actions': WasteIntelligenceEngine.ACTION_RECOMMENDATIONS[predicted_class],
            'class_probabilities': {f'Cat {i+1}': round(float(class_probs[i]), 4) for i in range(5)}
        }

print(" Waste Intelligence Engine initialized!")
print("   Capabilities: Waste Segregation, Pollution Scoring, Impact Assessment, Action Plans")

def full_inference(image_path, model, device, transform):
    """
    Run complete inference pipeline on a single image.
    Returns comprehensive analysis with visualizations.
    """
    model.eval()
    
    # Load and preprocess
    original_img = Image.open(image_path).convert('RGB')
    img_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits, embeddings = model(img_tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))
    
    # Analyze
    analysis = WasteIntelligenceEngine.analyze(predicted_class, confidence, probs)
    
    return original_img, analysis

def visualize_analysis(original_img, analysis):
    """
    Create comprehensive visualization dashboard for a single image analysis.
    """
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)
    
    # --- Original Image ---
    ax_img = fig.add_subplot(gs[0:2, 0:2])
    ax_img.imshow(original_img.resize((400, 400)))
    color_map = {0: '#FF4444', 1: '#FF8800', 2: '#FFCC00', 3: '#44BB44', 4: '#2288FF'}
    border_color = color_map[analysis['predicted_class']]
    for spine in ax_img.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(4)
    ax_img.set_title(f" {analysis['category_name']}\n"
                     f"Confidence: {analysis['confidence']*100:.1f}%",
                     fontsize=14, fontweight='bold', color=border_color)
    ax_img.axis('off')
    
    # --- Class Probabilities ---
    ax_prob = fig.add_subplot(gs[0, 2:4])
    cats = list(analysis['class_probabilities'].keys())
    vals = list(analysis['class_probabilities'].values())
    colors = ['#FF4444', '#FF8800', '#FFCC00', '#44BB44', '#2288FF']
    bars = ax_prob.barh(cats, vals, color=colors, edgecolor='white')
    ax_prob.set_title('Class Probabilities', fontsize=13, fontweight='bold')
    ax_prob.set_xlim(0, 1)
    for bar, val in zip(bars, vals):
        ax_prob.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2.,
                     f'{val*100:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # --- Waste Composition Pie Chart ---
    ax_waste = fig.add_subplot(gs[1, 2:4])
    wc = analysis['waste_composition']
    w_labels = [f"{k}\n{v*100:.1f}%" for k, v in wc.items() if v > 0.01]
    w_vals = [v for v in wc.values() if v > 0.01]
    w_colors = ['#3498db', '#2ecc71', '#95a5a6', '#e67e22', '#e74c3c', '#9b59b6'][:len(w_vals)]
    wedges, texts = ax_waste.pie(w_vals, labels=w_labels, colors=w_colors,
                                 startangle=90, textprops={'fontsize': 9})
    ax_waste.set_title('Waste Type Composition', fontsize=13, fontweight='bold')
    
    # --- Pollution Score Gauge ---
    ax_gauge = fig.add_subplot(gs[2, 0])
    score = analysis['pollution_score']
    gauge_color = '#FF4444' if score >= 7 else ('#FFAA00' if score >= 4 else '#44BB44')
    ax_gauge.barh(['Pollution'], [score], color=gauge_color, height=0.5, edgecolor='white')
    ax_gauge.barh(['Pollution'], [10], color='#f0f0f0', height=0.5, zorder=0)
    ax_gauge.set_xlim(0, 10)
    ax_gauge.set_title(f'Pollution Score: {score}/10\n({analysis["pollution_severity"]} — {analysis["pollution_level"]})',
                       fontsize=12, fontweight='bold', color=gauge_color)
    ax_gauge.set_yticks([])
    ax_gauge.text(score, 0, f' {score}', va='center', fontsize=14, fontweight='bold', color=gauge_color)
    
    # --- Impact Assessment ---
    ax_impact = fig.add_subplot(gs[2, 1:3])
    ax_impact.axis('off')
    impact_text = "\n".join([f"• {k.title()}: {v}" for k, v in analysis['impact'].items()])
    ax_impact.text(0, 1, ' Pollution Impact Assessment:', fontsize=13, fontweight='bold',
                   va='top', transform=ax_impact.transAxes)
    ax_impact.text(0, 0.85, impact_text, fontsize=9, va='top',
                   transform=ax_impact.transAxes, wrap=True,
                   fontfamily='monospace')
    
    # --- Actions ---
    ax_actions = fig.add_subplot(gs[2, 3])
    ax_actions.axis('off')
    actions_text = "\n".join(analysis['actions'][:5])
    ax_actions.text(0, 1, ' Recommended Actions:', fontsize=13, fontweight='bold',
                    va='top', transform=ax_actions.transAxes)
    ax_actions.text(0, 0.85, actions_text, fontsize=8, va='top',
                    transform=ax_actions.transAxes, wrap=True)
    
    plt.suptitle(' Neural Nexus — Waste Intelligence Report', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig

# ===== Run inference on sample images from each category =====
print("\n Running inference on sample images...\n")

for cat_idx in range(5):
    cat_images = [X_test[i] for i in range(len(X_test)) if y_test[i] == cat_idx]
    if cat_images:
        sample_path = random.choice(cat_images)
        try:
            original_img, analysis = full_inference(sample_path, model, device, val_transform)
            fig = visualize_analysis(original_img, analysis)
            fig.savefig(f'd:\\NEURAL NEXUS\\inference_cat{cat_idx+1}.png', dpi=100, bbox_inches='tight')
            plt.show()
            plt.close()
            
            print(f"Category {cat_idx+1} Sample:")
            print(f"  Predicted: {analysis['category_name']}")
            print(f"  Confidence: {analysis['confidence']*100:.1f}%")
            print(f"  Pollution Score: {analysis['pollution_score']}/10 ({analysis['pollution_level']})")
            wc = analysis['waste_composition']
            print(f"  Waste: Plastic={wc['Plastic']*100:.1f}%, Organic={wc['Organic']*100:.1f}%, "
                  f"Metallic={wc['Metallic']*100:.1f}%, E-waste={wc['E-waste']*100:.1f}%, "
                  f"Hazardous={wc['Hazardous']*100:.1f}%")
            print()
        except Exception as e:
            print(f"  Error processing image: {e}")

# Aggregate analysis across entire test set
all_analyses = []
all_pollution_scores = []
all_waste_compositions = {wtype: [] for wtype in WasteIntelligenceEngine.WASTE_TYPES}

for i in range(len(test_preds)):
    analysis = WasteIntelligenceEngine.analyze(
        int(test_preds[i]), 
        float(max_probs[i]),
        test_probs[i]
    )
    all_analyses.append(analysis)
    all_pollution_scores.append(analysis['pollution_score'])
    for wtype, pct in analysis['waste_composition'].items():
        all_waste_compositions[wtype].append(pct)

# ===== Aggregate Metrics =====
print("=" * 80)
print(" AGGREGATE TEST SET ANALYSIS")
print("=" * 80)

print(f"\n Model Performance Summary:")
print(f"   ┌─────────────────────────────────┐")
print(f"   │ Accuracy:     {test_accuracy*100:6.2f}%           │")
print(f"   │ F1 Score:     {test_f1*100:6.2f}%           │")
print(f"   │ Precision:    {test_precision*100:6.2f}%           │")
print(f"   │ Recall:       {test_recall*100:6.2f}%           │")
print(f"   │ Confidence:   {mean_confidence*100:6.2f}%           │")
print(f"   └─────────────────────────────────┘")

print(f"\n Pollution Analysis (Test Set):")
print(f"   Mean Pollution Score: {np.mean(all_pollution_scores):.1f}/10")
print(f"   Median Pollution Score: {np.median(all_pollution_scores):.1f}/10")
print(f"   Std Deviation: {np.std(all_pollution_scores):.1f}")

# Pollution level breakdown
high_count = sum(1 for s in all_pollution_scores if s >= 7)
medium_count = sum(1 for s in all_pollution_scores if 4 <= s < 7)
low_count = sum(1 for s in all_pollution_scores if s < 4)
total_t = len(all_pollution_scores)

print(f"\n   Pollution Level Breakdown:")
print(f"   High (7-10):   {high_count} ({high_count/total_t*100:.1f}%)")
print(f"   🟡 Medium (4-7):  {medium_count} ({medium_count/total_t*100:.1f}%)")
print(f"   🟢 Low (0-4):     {low_count} ({low_count/total_t*100:.1f}%)")

print(f"\n Average Waste Composition (Test Set):")
for wtype in WasteIntelligenceEngine.WASTE_TYPES:
    avg_pct = np.mean(all_waste_compositions[wtype]) * 100
    bar = '█' * int(avg_pct / 2)
    emoji = {'Plastic': '', 'Organic': '', 'Metallic': '', 
             'E-waste': '', 'Hazardous': '', 'Anomalies': ''}[wtype]
    print(f"   {emoji} {wtype:12s}: {avg_pct:5.1f}% {bar}")

fig = plt.figure(figsize=(24, 16))
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)

# --- 1. Pollution Score Distribution ---
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.hist(all_pollution_scores, bins=20, color='#e74c3c', edgecolor='white', alpha=0.8)
ax1.axvline(x=np.mean(all_pollution_scores), color='blue', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(all_pollution_scores):.1f}')
ax1.set_title('Pollution Score Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Pollution Score (0-10)')
ax1.set_ylabel('Count')
ax1.legend(fontsize=12)

# --- 2. Waste Composition Breakdown ---
ax2 = fig.add_subplot(gs[0, 2:4])
waste_means = {wtype: np.mean(vals)*100 for wtype, vals in all_waste_compositions.items()}
w_colors = ['#3498db', '#2ecc71', '#95a5a6', '#e67e22', '#e74c3c', '#9b59b6']
bars = ax2.bar(waste_means.keys(), waste_means.values(), color=w_colors, edgecolor='white')
ax2.set_title('Average Waste Composition', fontsize=14, fontweight='bold')
ax2.set_ylabel('Percentage (%)')
for bar, val in zip(bars, waste_means.values()):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax2.set_xticklabels(waste_means.keys(), rotation=15)

# --- 3. Confidence vs Accuracy per Category ---
ax3 = fig.add_subplot(gs[1, 0:2])
for i in range(5):
    mask = test_targets == i
    if mask.sum() > 0:
        cat_probs = max_probs[mask]
        cat_correct = (test_preds[mask] == i)
        ax3.scatter(cat_probs[cat_correct], [i+1]*cat_correct.sum(), 
                    alpha=0.3, s=20, color=colors[i], label=f'Cat {i+1} ')
        ax3.scatter(cat_probs[~cat_correct], [i+1]*(~cat_correct).sum(),
                    alpha=0.3, s=20, color='red', marker='x')
ax3.set_title('Confidence Distribution per Category', fontsize=14, fontweight='bold')
ax3.set_xlabel('Prediction Confidence')
ax3.set_ylabel('Category')
ax3.set_yticks([1, 2, 3, 4, 5])
ax3.set_yticklabels(['Cat 1', 'Cat 2', 'Cat 3', 'Cat 4', 'Cat 5'])

# --- 4. Pollution Level Pie ---
ax4 = fig.add_subplot(gs[1, 2])
p_labels = ['High', 'Medium', 'Low']
p_values = [high_count, medium_count, low_count]
p_colors = ['#e74c3c', '#f39c12', '#27ae60']
wedges, _, autotexts = ax4.pie(p_values, labels=p_labels, colors=p_colors,
                                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
ax4.set_title('Pollution Level\nBreakdown', fontsize=13, fontweight='bold')

# --- 5. Model Metrics Summary ---
ax5 = fig.add_subplot(gs[1, 3])
ax5.axis('off')
metrics_text = (
    f" MODEL METRICS\n"
    f"{'─'*30}\n"
    f"Accuracy:    {test_accuracy*100:.2f}%\n"
    f"F1 Score:    {test_f1*100:.2f}%\n"
    f"Precision:   {test_precision*100:.2f}%\n"
    f"Recall:      {test_recall*100:.2f}%\n"
    f"{'─'*30}\n"
    f" CONFIDENCE\n"
    f"{'─'*30}\n"
    f"Mean:        {mean_confidence*100:.2f}%\n"
    f"Correct:     {correct_confidence*100:.2f}%\n"
    f"Incorrect:   {incorrect_confidence*100:.2f}%\n"
    f"{'─'*30}\n"
    f" POLLUTION\n"
    f"{'─'*30}\n"
    f"Avg Score:   {np.mean(all_pollution_scores):.1f}/10\n"
    f"Total Images: {len(test_preds)}"
)
ax5.text(0.05, 0.95, metrics_text, fontsize=11, va='top', transform=ax5.transAxes,
         fontfamily='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.8))

# --- 6. Waste Composition per Category (Stacked Bar) ---
ax6 = fig.add_subplot(gs[2, 0:2])
cat_waste = {}
for i in range(5):
    mask = test_preds == i
    cat_waste[f'Cat {i+1}'] = {}
    for wtype in WasteIntelligenceEngine.WASTE_TYPES:
        if mask.sum() > 0:
            cat_waste[f'Cat {i+1}'][wtype] = np.mean([all_waste_compositions[wtype][j] 
                                                       for j in range(len(test_preds)) if test_preds[j] == i]) * 100
        else:
            cat_waste[f'Cat {i+1}'][wtype] = 0

cat_labels = list(cat_waste.keys())
bottom = np.zeros(5)
for idx, wtype in enumerate(WasteIntelligenceEngine.WASTE_TYPES):
    values = [cat_waste[cat][wtype] for cat in cat_labels]
    ax6.bar(cat_labels, values, bottom=bottom, label=wtype, color=w_colors[idx], edgecolor='white')
    bottom += values

ax6.set_title('Waste Composition per Category', fontsize=14, fontweight='bold')
ax6.set_ylabel('Percentage (%)')
ax6.legend(loc='upper right', fontsize=9)

# --- 7. Prediction Confidence Box Plot ---
ax7 = fig.add_subplot(gs[2, 2:4])
conf_by_cat = [max_probs[test_targets == i] for i in range(5)]
bp = ax7.boxplot(conf_by_cat, labels=class_names, patch_artist=True, 
                  notch=True, showmeans=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax7.set_title('Confidence Distribution by Category', fontsize=14, fontweight='bold')
ax7.set_ylabel('Prediction Confidence')

plt.suptitle(' Neural Nexus — Complete Intelligence Dashboard', fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'd:\NEURAL NEXUS\complete_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n Complete dashboard saved to: d:\\NEURAL NEXUS\\complete_dashboard.png")

class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Forward pass
        output, _ = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Compute weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # Compute CAM
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class

# Initialize Grad-CAM with last conv layer
target_layer = model.backbone.features[-1]
grad_cam = GradCAM(model, target_layer)

# Generate Grad-CAM for sample images
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

for cat_idx in range(5):
    cat_images = [X_test[i] for i in range(len(X_test)) if y_test[i] == cat_idx]
    if cat_images:
        sample_path = random.choice(cat_images)
        try:
            original_img = Image.open(sample_path).convert('RGB')
            img_resized = original_img.resize((224, 224))
            img_tensor = val_transform(original_img).unsqueeze(0).to(device)
            
            # Generate attention map
            cam, pred_class = grad_cam.generate(img_tensor)
            
            # Original image
            axes[0, cat_idx].imshow(img_resized)
            axes[0, cat_idx].set_title(f'Cat {cat_idx+1} (True)', fontsize=12, fontweight='bold')
            axes[0, cat_idx].axis('off')
            
            # Attention map overlay
            axes[1, cat_idx].imshow(img_resized)
            axes[1, cat_idx].imshow(cam, alpha=0.5, cmap='jet')
            axes[1, cat_idx].set_title(f'Pred: Cat {pred_class+1} (Grad-CAM)', fontsize=12, fontweight='bold')
            axes[1, cat_idx].axis('off')
        except Exception as e:
            axes[0, cat_idx].text(0.5, 0.5, f'Error', ha='center', va='center')
            axes[1, cat_idx].text(0.5, 0.5, f'Error', ha='center', va='center')
            axes[0, cat_idx].axis('off')
            axes[1, cat_idx].axis('off')

plt.suptitle(' Grad-CAM Attention Maps — Where the Model Looks', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(r'd:\NEURAL NEXUS\gradcam_attention.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n Grad-CAM attention maps generated!")
print("   Red/Yellow areas = High attention (model focuses here)")
print("   Blue/Green areas = Low attention")

# Save model
save_path = r'd:\NEURAL NEXUS\neural_nexus_model_final.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': 'EfficientNet-B0 + Custom Head',
    'num_classes': 5,
    'input_size': 224,
    'accuracy': test_accuracy,
    'f1_score': test_f1,
    'mean_confidence': mean_confidence,
    'training_epochs': len(history['train_loss']),
    'class_names': list(CATEGORY_MAP.values()),
}, save_path)

print(f" Model saved to: {save_path}")

# Save training history
history_path = r'd:\NEURAL NEXUS\training_history.json'
with open(history_path, 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)
print(f" Training history saved to: {history_path}")

# Final comprehensive report
report = {
    'model': {
        'architecture': 'EfficientNet-B0 + Custom Multi-Task Head',
        'parameters': f'{total_params:,}',
        'trainable_parameters': f'{trainable_params:,}',
        'input_size': '224x224',
        'backbone': 'EfficientNet-B0 (ImageNet pretrained)',
    },
    'performance': {
        'accuracy': f'{test_accuracy*100:.2f}%',
        'f1_score': f'{test_f1*100:.2f}%',
        'precision': f'{test_precision*100:.2f}%',
        'recall': f'{test_recall*100:.2f}%',
        'mean_confidence': f'{mean_confidence*100:.2f}%',
        'correct_prediction_confidence': f'{correct_confidence*100:.2f}%',
    },
    'pollution_analysis': {
        'mean_pollution_score': f'{np.mean(all_pollution_scores):.1f}/10',
        'high_pollution_percentage': f'{high_count/total_t*100:.1f}%',
        'medium_pollution_percentage': f'{medium_count/total_t*100:.1f}%',
        'low_pollution_percentage': f'{low_count/total_t*100:.1f}%',
    },
    'waste_composition': {wtype: f'{np.mean(vals)*100:.1f}%' 
                          for wtype, vals in all_waste_compositions.items()},
    'dataset': {
        'total_images': len(all_images),
        'train_images': len(X_train),
        'val_images': len(X_val),
        'test_images': len(X_test),
    },
    'training': {
        'epochs_trained': len(history['train_loss']),
        'optimizer': 'AdamW',
        'loss_function': 'Label Smoothing Cross Entropy',
        'augmentations': 'Extensive (10 techniques)',
        'class_balancing': 'WeightedRandomSampler + Cost-sensitive learning',
    }
}

report_path = r'd:\NEURAL NEXUS\final_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)
print(f" Final report saved to: {report_path}")

# Print final summary
print("\n" + "=" * 80)
print(" NEURAL NEXUS — FINAL SYSTEM REPORT")
print("=" * 80)
print(f"\n Architecture: EfficientNet-B0 + Custom Head ({total_params:,} params)")
print(f"\n Performance Metrics:")
print(f"   ╔══════════════════════════════════╗")
print(f"   ║  Accuracy:     {test_accuracy*100:6.2f}%            ║")
print(f"   ║  F1 Score:     {test_f1*100:6.2f}%            ║")
print(f"   ║  Precision:    {test_precision*100:6.2f}%            ║")
print(f"   ║  Recall:       {test_recall*100:6.2f}%            ║")
print(f"   ║  Confidence:   {mean_confidence*100:6.2f}%            ║")
print(f"   ╚══════════════════════════════════╝")
print(f"\n Pollution Summary:")
print(f"   Avg Score: {np.mean(all_pollution_scores):.1f}/10")
print(f"   High: {high_count/total_t*100:.1f}% | 🟡 Medium: {medium_count/total_t*100:.1f}% | 🟢 Low: {low_count/total_t*100:.1f}%")
print(f"\n Waste Breakdown:")
for wtype in WasteIntelligenceEngine.WASTE_TYPES:
    pct = np.mean(all_waste_compositions[wtype]) * 100
    emoji = {'Plastic': '', 'Organic': '', 'Metallic': '', 
             'E-waste': '', 'Hazardous': '', 'Anomalies': ''}[wtype]
    print(f"   {emoji} {wtype:12s}: {pct:5.1f}%")

print(f"\n Saved Artifacts:")
print(f"   • Model: {save_path}")
print(f"   • Training History: {history_path}")
print(f"   • Final Report: {report_path}")
print(f"   • Training Curves: d:\\NEURAL NEXUS\\training_curves.png")
print(f"   • Evaluation Results: d:\\NEURAL NEXUS\\evaluation_results.png")
print(f"   • Complete Dashboard: d:\\NEURAL NEXUS\\complete_dashboard.png")
print(f"   • Grad-CAM Maps: d:\\NEURAL NEXUS\\gradcam_attention.png")
print(f"\n{'='*80}")
print(" Neural Nexus Waste Intelligence System — Training & Evaluation Complete!")
print("='*80")

def predict_waste(image_path):
    """
    Run complete waste intelligence prediction on any image.
    
    Usage:
        predict_waste(r'd:\path\to\your\image.jpg')
    """
    if not os.path.exists(image_path):
        print(f" Image not found: {image_path}")
        return
    
    try:
        original_img, analysis = full_inference(image_path, model, device, val_transform)
        fig = visualize_analysis(original_img, analysis)
        plt.show()
        plt.close()
        
        print(f"\n{'='*60}")
        print(f" PREDICTION SUMMARY")
        print(f"{'='*60}")
        print(f"  Classification:  {analysis['category_name']}")
        print(f"  Confidence:      {analysis['confidence']*100:.1f}%")
        print(f"  Pollution Score: {analysis['pollution_score']}/10 ({analysis['pollution_level']})")
        print(f"  Severity:        {analysis['pollution_severity']}")
        print(f"\n  Waste Composition:")
        for wtype, pct in analysis['waste_composition'].items():
            if pct > 0.01:
                print(f"     {wtype:12s}: {pct*100:.1f}%")
        print(f"\n  Top Actions:")
        for action in analysis['actions'][:3]:
            print(f"     {action}")
        return analysis
    except Exception as e:
        print(f" Error: {e}")
        return None

# Example usage: predict on a random test image
sample_path = random.choice(X_test)
print(f" Running prediction on: {os.path.basename(sample_path)}\n")
result = predict_waste(sample_path)





