"""
Script de Fine-tuning CRNN pour plaques tunisiennes
Entraîne un modèle de reconnaissance de texte sur le dataset de plaques
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import string
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
DATASET_PATH = Path(r"C:\Users\Malek\Desktop\AutoPlateTN\data\raw\OCR_Dataset_Models")
IMAGES_PATH = DATASET_PATH / "license_plates_recognition_train" / "license_plates_recognition_train"
CSV_PATH = DATASET_PATH / "license_plates_recognition_train.csv"
OUTPUT_PATH = DATASET_PATH / "crnn_model"
OUTPUT_PATH.mkdir(exist_ok=True)

# Hyperparamètres
IMG_HEIGHT = 32
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Caractères supportés (chiffres + T pour "تونس")
CHARACTERS = string.digits + 'T'  # 0-9 + T
NUM_CLASSES = len(CHARACTERS) + 1  # +1 pour blank (CTC)

print(f"✓ Caractères supportés: {CHARACTERS}")
print(f"✓ Nombre de classes: {NUM_CLASSES}")


class TunisianPlateDataset(Dataset):
    """Dataset pour plaques tunisiennes"""
    
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Filtre pour garder seulement les images existantes
        self.data = self.data[self.data['img_id'].apply(
            lambda x: (self.img_dir / x).exists()
        )].reset_index(drop=True)
        
        print(f"✓ Dataset chargé: {len(self.data)} images")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Charger l'image
        img_name = self.data.loc[idx, 'img_id']
        img_path = self.img_dir / img_name
        image = cv2.imread(str(img_path))
        
        if image is None:
            # Image de fallback si erreur
            image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
        
        # Convertir BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        
        # Appliquer transforms
        if self.transform:
            image = self.transform(image)
        
        # Label (texte)
        label = self.data.loc[idx, 'text']
        
        # Encoder le texte en indices
        encoded_label = self.encode_text(label)
        
        return image, encoded_label, len(encoded_label)
    
    def encode_text(self, text):
        """Encode le texte en indices de caractères"""
        encoded = []
        for char in text:
            if char in CHARACTERS:
                encoded.append(CHARACTERS.index(char) + 1)  # +1 car 0 = blank
        return encoded


class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) pour reconnaissance de texte
    Architecture: CNN -> RNN (LSTM) -> Fully Connected
    """
    
    def __init__(self, img_height, num_classes, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()
        
        self.img_height = img_height
        self.num_classes = num_classes
        
        # CNN Backbone (VGG-like)
        self.cnn = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 32x128 -> 16x64
            
            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 16x64 -> 8x32
            
            # Conv3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Conv4
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 8x32 -> 4x32 (hauteur seulement)
            
            # Conv5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Conv6
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 4x32 -> 2x32
            
            # Conv7
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),  # 2x32 -> 1x31
        )
        
        # RNN (LSTM bidirectionnel)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Fully Connected
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 car bidirectionnel
    
    def forward(self, x):
        # CNN
        conv = self.cnn(x)  # (batch, 512, 1, width)
        
        # Reshape pour RNN: (batch, width, features)
        batch, channels, height, width = conv.size()
        conv = conv.squeeze(2)  # (batch, 512, width)
        conv = conv.permute(0, 2, 1)  # (batch, width, 512)
        
        # RNN
        rnn_out, _ = self.rnn(conv)  # (batch, width, hidden_size*2)
        
        # FC
        output = self.fc(rnn_out)  # (batch, width, num_classes)
        
        # Pour CTC: (width, batch, num_classes)
        output = output.permute(1, 0, 2)
        
        return output


def collate_fn(batch):
    """Fonction de collation pour gérer des longueurs de texte variables"""
    images, labels, lengths = zip(*batch)
    
    # Stack images
    images = torch.stack([img for img in images], 0)
    
    # Concaténer tous les labels
    labels = [torch.LongTensor(label) for label in labels]
    labels = torch.cat(labels)
    
    # Longueurs
    lengths = torch.LongTensor(lengths)
    
    return images, labels, lengths


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîner une époque"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels, label_lengths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)
        
        # Forward
        outputs = model(images)  # (T, N, C)
        
        # Longueurs des séquences de sortie (toutes identiques)
        T, N, C = outputs.size()
        input_lengths = torch.full((N,), T, dtype=torch.long, device=device)
        
        # CTC Loss
        loss = criterion(
            outputs.log_softmax(2),
            labels,
            input_lengths,
            label_lengths
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Valider le modèle"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, labels, label_lengths in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            
            # Forward
            outputs = model(images)
            
            # Longueurs
            T, N, C = outputs.size()
            input_lengths = torch.full((N,), T, dtype=torch.long, device=device)
            
            # Loss
            loss = criterion(
                outputs.log_softmax(2),
                labels,
                input_lengths,
                label_lengths
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def decode_predictions(outputs):
    """Décoder les prédictions CTC"""
    # outputs: (T, N, C)
    _, preds = outputs.max(2)  # (T, N)
    preds = preds.transpose(1, 0).contiguous()  # (N, T)
    
    decoded = []
    for pred in preds:
        # CTC beam search simple (greedy)
        chars = []
        prev_char = -1
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev_char:  # Pas blank et pas répétition
                chars.append(CHARACTERS[idx - 1])
            prev_char = idx
        decoded.append(''.join(chars))
    
    return decoded


def main():
    """Fonction principale d'entraînement"""
    print("="*60)
    print("FINE-TUNING CRNN POUR PLAQUES TUNISIENNES")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    print("\n📁 Chargement du dataset...")
    full_dataset = TunisianPlateDataset(CSV_PATH, IMAGES_PATH, transform=transform)
    
    # Split train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"✓ Train: {len(train_dataset)} images")
    print(f"✓ Val: {len(val_dataset)} images")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Windows: 0 workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Modèle
    print("\n🔧 Initialisation du modèle CRNN...")
    model = CRNN(IMG_HEIGHT, NUM_CLASSES, hidden_size=256, num_layers=2)
    model = model.to(device)
    
    # Compter les paramètres
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Nombre de paramètres: {num_params:,}")
    
    # Loss & Optimizer
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print(f"\n🚀 Début de l'entraînement ({EPOCHS} époques)...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}")
        print(f"Époque {epoch+1}/{EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        print(f"✓ Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"✓ Val Loss: {val_loss:.4f}")
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, OUTPUT_PATH / 'best_crnn_model.pth')
            print(f"💾 Meilleur modèle sauvegardé (val_loss: {val_loss:.4f})")
        
        # Save checkpoint régulier
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, OUTPUT_PATH / f'checkpoint_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save(model.state_dict(), OUTPUT_PATH / 'final_crnn_model.pth')
    print(f"\n✅ Entraînement terminé!")
    print(f"✅ Modèle final sauvegardé: {OUTPUT_PATH / 'final_crnn_model.pth'}")
    
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.title('Courbes d\'entraînement')
    plt.legend()
    plt.grid(True)
    plt.savefig(OUTPUT_PATH / 'training_curves.png')
    print(f"📊 Courbes sauvegardées: {OUTPUT_PATH / 'training_curves.png'}")
    
    # Test sur quelques exemples
    print("\n🔍 Test sur quelques exemples:")
    model.eval()
    with torch.no_grad():
        for i, (images, labels, label_lengths) in enumerate(val_loader):
            if i >= 3:  # 3 batchs
                break
            
            images = images.to(device)
            outputs = model(images)
            predictions = decode_predictions(outputs)
            
            # Décoder les vrais labels
            start = 0
            true_labels = []
            for length in label_lengths:
                label_seq = labels[start:start+length]
                true_text = ''.join([CHARACTERS[idx-1] for idx in label_seq])
                true_labels.append(true_text)
                start += length
            
            # Afficher
            for pred, true in zip(predictions[:5], true_labels[:5]):
                print(f"  Prédit: {pred:12s} | Vrai: {true}")


if __name__ == "__main__":
    main()
