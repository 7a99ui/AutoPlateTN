"""
Pipeline OCR Testing Script
Tests all 4 models (original, light, full, aggressive) on detected plates
Each plate has 4 enhanced versions, tests with corresponding model
"""
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Get the base directory (where this script is located)
BASE_DIR = Path(__file__).parent.absolute()

# Configuration (relative paths)
MODELS_BASE = BASE_DIR / "models" / "ocr"
DETECTED_PLATES_DIR = BASE_DIR / "data" / "processed" / "detected_plates"
ENHANCED_BASE_DIR = BASE_DIR / "data" / "processed" / "enhanced_plates"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "ocr_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model Configuration
IMG_HEIGHT = 32
IMG_WIDTH = 128
ENHANCEMENT_METHODS = ['original', 'light', 'full', 'aggressive']

# Characters (will be loaded from model)
CHARACTERS = '0123456789T'
NUM_CLASSES = len(CHARACTERS) + 1


class CRNN(torch.nn.Module):
    """CRNN model architecture"""
    def __init__(self, img_height, num_classes, hidden_size=256, num_layers=2):
        super(CRNN, self).__init__()
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 1), (2, 1)),
            
            torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d((2, 1), (2, 1)),
            
            torch.nn.Conv2d(512, 512, kernel_size=2, padding=0),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True)
        )
        
        self.rnn = torch.nn.LSTM(512, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)
        output = output.permute(1, 0, 2)
        return output


class PlateRecognizer:
    """Plate recognizer with specific model"""
    
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        
        # Load checkpoint to get characters
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if 'characters' in checkpoint:
            global CHARACTERS, NUM_CLASSES
            CHARACTERS = checkpoint['characters']
            NUM_CLASSES = len(CHARACTERS) + 1
        
        # Initialize model
        self.model = CRNN(IMG_HEIGHT, NUM_CLASSES, hidden_size=256, num_layers=2)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image_path):
        """Preprocess image"""
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot load: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor
    
    def decode(self, output):
        """Decode CTC prediction"""
        _, pred = output.max(2)
        pred = pred.squeeze(1)
        
        chars = []
        prev_char = -1
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev_char:
                if idx - 1 < len(CHARACTERS):
                    chars.append(CHARACTERS[idx - 1])
            prev_char = idx
        
        return ''.join(chars)
    
    def format_plate_text(self, raw_text):
        """Format plate text with Arabic"""
        digits = ''.join(c for c in raw_text if c.isdigit())
        
        if 'T' in raw_text.upper():
            # Tunisia plate
            if len(digits) >= 4:
                t_pos = raw_text.upper().find('T')
                left_chars = sum(1 for c in raw_text[:t_pos] if c.isdigit())
                
                if left_chars > 0:
                    left = digits[:left_chars]
                    right = digits[left_chars:]
                    return f"{left} تونس {right}"
                else:
                    if len(digits) >= 7:
                        left = digits[:3]
                        right = digits[3:]
                    else:
                        mid = len(digits) // 2
                        left = digits[:mid]
                        right = digits[mid:]
                    return f"{left} تونس {right}"
            else:
                return f"{digits} تونس"
        
        elif 'N' in raw_text.upper():
            # Private plate
            if len(digits) >= 6:
                left = digits[:6]
                return f"{left} نت"
            else:
                return f"{digits} نت"
        
        else:
            # No marker, guess based on length
            if len(digits) >= 7:
                left = digits[:3]
                right = digits[3:]
                return f"{left} تونس {right}"
            elif len(digits) == 6:
                return f"{digits} نت"
            else:
                return raw_text
    
    def recognize(self, image_path):
        """Recognize plate text"""
        image_tensor = self.preprocess(image_path).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        raw_text = self.decode(output)
        formatted_text = self.format_plate_text(raw_text)
        
        return {
            'raw': raw_text,
            'formatted': formatted_text
        }


def get_detected_plate_names(detected_dir):
    """Get all base names of detected plates"""
    plates = []
    for file in os.listdir(detected_dir):
        if file.endswith(('.jpg', '.jpeg', '.png')):
            # Extract base name (e.g., "plate_001.jpg" -> "plate_001")
            base_name = os.path.splitext(file)[0]
            plates.append({
                'base_name': base_name,
                'original_file': file
            })
    return plates


def process_all_plates(detected_dir, enhanced_base_dir, models_base, output_dir, device):
    """Process all detected plates with all 4 models"""
    
    print(f"\n{'='*70}")
    print("PIPELINE OCR - TESTING ALL 4 MODELS ON DETECTED PLATES")
    print(f"{'='*70}\n")
    
    # Load all 4 models
    models = {}
    print("Loading models...")
    
    for method in ENHANCEMENT_METHODS:
        if method == 'original':
            model_path = models_base / "best_crnn_model.pth"
        else:
            model_path = models_base / method / f"best_crnn_{method}.pth"
        
        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            continue
        
        models[method] = PlateRecognizer(str(model_path), device=device)
        print(f"  ✓ Loaded {method} model")
    
    if not models:
        print("❌ No models loaded!")
        return
    
    print(f"\n✓ Loaded {len(models)} models\n")
    
    # Get all detected plates
    plates = get_detected_plate_names(detected_dir)
    
    if not plates:
        print(f"❌ No detected plates found in: {detected_dir}")
        return
    
    print(f"Found {len(plates)} detected plates\n")
    print(f"{'='*70}\n")
    
    # Process each plate
    results = []
    
    for plate_info in tqdm(plates, desc="Processing plates"):
        base_name = plate_info['base_name']
        original_file = plate_info['original_file']
        
        print(f"\n{'─'*70}")
        print(f"Plate: {base_name}")
        print(f"{'─'*70}")
        
        plate_result = {
            'base_name': base_name,
            'original_file': original_file,
            'methods': {}
        }
        
        # Test with each model on its corresponding enhanced image
        for method in ENHANCEMENT_METHODS:
            if method not in models:
                continue
            
            # Get the enhanced image path
            if method == 'original':
                # Use the original detected plate
                image_path = detected_dir / original_file
            else:
                # Use the enhanced version
                enhanced_file = f"{base_name}_{method}.jpg"
                image_path = enhanced_base_dir / method / enhanced_file
                
                # Try .png if .jpg doesn't exist
                if not image_path.exists():
                    enhanced_file = f"{base_name}_{method}.png"
                    image_path = enhanced_base_dir / method / enhanced_file
            
            if not image_path.exists():
                plate_result['methods'][method] = {
                    'raw': '',
                    'formatted': '',
                    'error': 'Image not found',
                    'image_path': str(image_path)
                }
                print(f"  {method:12s}: ❌ Image not found")
                continue
            
            try:
                # Run OCR
                result = models[method].recognize(image_path)
                plate_result['methods'][method] = {
                    'raw': result['raw'],
                    'formatted': result['formatted'],
                    'error': None,
                    'image_path': str(image_path)
                }
                print(f"  {method:12s}: {result['raw']:15s} → {result['formatted']}")
                
            except Exception as e:
                plate_result['methods'][method] = {
                    'raw': '',
                    'formatted': '',
                    'error': str(e),
                    'image_path': str(image_path)
                }
                print(f"  {method:12s}: ❌ Error - {e}")
        
        results.append(plate_result)
    
    # Save results
    save_results(results, output_dir)
    
    return results


def save_results(results, output_dir):
    """Save OCR results"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}\n")
    
    # 1. CSV - Comparison table
    csv_data = []
    for result in results:
        row = {'Plate': result['base_name']}
        for method in ENHANCEMENT_METHODS:
            if method in result['methods']:
                row[method.capitalize()] = result['methods'][method]['formatted']
            else:
                row[method.capitalize()] = 'N/A'
        csv_data.append(row)
    
    csv_path = output_dir / f"ocr_comparison_{timestamp}.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ CSV comparison: {csv_path}")
    
    # 2. Detailed CSV with raw text
    detailed_data = []
    for result in results:
        for method in ENHANCEMENT_METHODS:
            if method in result['methods']:
                detailed_data.append({
                    'Plate': result['base_name'],
                    'Method': method.capitalize(),
                    'Raw_Text': result['methods'][method]['raw'],
                    'Formatted_Text': result['methods'][method]['formatted'],
                    'Status': 'Success' if not result['methods'][method]['error'] else 'Failed',
                    'Error': result['methods'][method].get('error', '')
                })
    
    detailed_csv_path = output_dir / f"ocr_detailed_{timestamp}.csv"
    pd.DataFrame(detailed_data).to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Detailed CSV: {detailed_csv_path}")
    
    # 3. Text summary
    txt_path = output_dir / f"ocr_summary_{timestamp}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("PIPELINE OCR RESULTS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total plates processed: {len(results)}\n\n")
        
        for result in results:
            f.write(f"\nPlate: {result['base_name']}\n")
            f.write("─"*70 + "\n")
            for method in ENHANCEMENT_METHODS:
                if method in result['methods']:
                    raw = result['methods'][method]['raw']
                    formatted = result['methods'][method]['formatted']
                    error = result['methods'][method]['error']
                    status = "✓" if not error else "✗"
                    f.write(f"  {status} {method:12s}: {raw:15s} → {formatted}\n")
            f.write("\n")
    
    print(f"✓ Text summary: {txt_path}")
    
    # 4. Statistics
    stats_path = output_dir / f"ocr_statistics_{timestamp}.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("OCR STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Total plates: {len(results)}\n\n")
        
        # Success rate per method
        f.write("Success rate by method:\n")
        for method in ENHANCEMENT_METHODS:
            success = sum(1 for r in results 
                         if method in r['methods'] and not r['methods'][method]['error'])
            total = sum(1 for r in results if method in r['methods'])
            rate = (success / total * 100) if total > 0 else 0
            f.write(f"  {method:12s}: {success}/{total} ({rate:.1f}%)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CONSENSUS ANALYSIS\n")
        f.write("="*70 + "\n\n")
        
        # Find plates where all methods agree
        all_agree = 0
        for result in results:
            texts = [r['formatted'] for r in result['methods'].values() if not r['error']]
            if len(texts) > 1 and len(set(texts)) == 1:
                all_agree += 1
        
        f.write(f"Plates where all methods agree: {all_agree}/{len(results)}\n")
        
    print(f"✓ Statistics: {stats_path}")
    
    print(f"\n{'='*70}")


def main():
    """Main function"""
    
    print("="*70)
    print("PIPELINE OCR - 4 MODELS ON DETECTED PLATES")
    print("="*70)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n✓ Device: {device}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    
    # Check paths
    print(f"\n📁 Checking paths...")
    print(f"  Models: {MODELS_BASE}")
    print(f"  Detected plates: {DETECTED_PLATES_DIR}")
    print(f"  Enhanced plates: {ENHANCED_BASE_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    
    if not DETECTED_PLATES_DIR.exists():
        print(f"\n❌ Detected plates directory not found!")
        print(f"   Please run plate detection first.")
        return
    
    if not ENHANCED_BASE_DIR.exists():
        print(f"\n❌ Enhanced plates directory not found!")
        print(f"   Please run enhancement first.")
        return
    
    # Process all plates
    try:
        results = process_all_plates(
            DETECTED_PLATES_DIR,
            ENHANCED_BASE_DIR,
            MODELS_BASE,
            OUTPUT_DIR,
            device
        )
        
        print(f"\n✅ PIPELINE OCR COMPLETE!")
        print(f"\n📊 Results saved in: {OUTPUT_DIR}")
        print(f"   - CSV comparison table (open in Excel)")
        print(f"   - Detailed CSV with raw text")
        print(f"   - Text summary")
        print(f"   - Statistics")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()