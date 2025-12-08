"""
License Plate Detection Script
Detects plates from all detected vehicles in the folder
Saves detected plates organized by vehicle
"""

import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import os
from pathlib import Path

# Get the base directory (where this script is located)
BASE_DIR = Path(__file__).parent.absolute()

# Configuration (relative paths)
VEHICLES_DIR = BASE_DIR / "data" / "processed" / "detected_vehicules"
OUTPUT_PLATES_DIR = BASE_DIR / "data" / "processed" / "detected_plates"
MODEL_PATH = BASE_DIR / "models" / "detection" / "fasterrcnn_tunisia_plates.pth"
SCORE_THRESHOLD = 0.5


def load_model(model_path):
    """Load the trained plate detection model"""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace predictor (background + license plate)
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features,
        num_classes=2
    )

    # Load saved weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    
    return model, device


def predict_plate(model, device, image, score_threshold=0.5):
    """
    Predict license plate location in the image
    
    Args:
        model: Trained plate detection model
        device: 'cuda' or 'cpu'
        image: OpenCV image (BGR format)
        score_threshold: Minimum confidence threshold
    
    Returns:
        List of cropped plates with their info
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor
    img_tensor = F.to_tensor(img_rgb).to(device)

    # Forward pass
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    boxes = predictions["boxes"].cpu()
    scores = predictions["scores"].cpu()

    # Filter by threshold
    valid_detections = []
    for box, score in zip(boxes, scores):
        if score >= score_threshold:
            x1, y1, x2, y2 = box.int().tolist()
            # Ensure valid coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 > x1 and y2 > y1:  # Valid box
                cropped_plate = image[y1:y2, x1:x2]
                valid_detections.append({
                    'box': [x1, y1, x2, y2],
                    'score': float(score),
                    'crop': cropped_plate
                })
    
    # Sort by score (highest first)
    valid_detections.sort(key=lambda x: x['score'], reverse=True)
    
    return valid_detections


def process_vehicles(model, device, vehicles_dir, output_dir, score_threshold):
    """
    Process all vehicle images and detect plates
    
    Args:
        model: Plate detection model
        device: 'cuda' or 'cpu'
        vehicles_dir: Directory containing detected vehicle images
        output_dir: Directory to save detected plates
        score_threshold: Minimum confidence threshold
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all vehicle images
    vehicle_files = sorted([f for f in os.listdir(vehicles_dir) 
                           if f.endswith(('.jpg', '.jpeg', '.png')) 
                           and not f.startswith('detection_visualization')])
    
    if len(vehicle_files) == 0:
        print(f"❌ No vehicle images found in: {vehicles_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Processing {len(vehicle_files)} vehicles for plate detection")
    print(f"{'='*70}\n")
    
    total_plates = 0
    vehicles_with_plates = 0
    
    for vehicle_file in vehicle_files:
        vehicle_path = vehicles_dir / vehicle_file
        
        # Load vehicle image
        vehicle_img = cv2.imread(str(vehicle_path))
        if vehicle_img is None:
            print(f"⚠️  Failed to load: {vehicle_file}")
            continue
        
        print(f"Processing: {vehicle_file}")
        
        # Detect plates
        detections = predict_plate(model, device, vehicle_img, score_threshold)
        
        if len(detections) == 0:
            print(f"  ⚠️  No plate detected")
            print()
            continue
        
        vehicles_with_plates += 1
        
        # Save all detected plates for this vehicle
        for idx, detection in enumerate(detections):
            plate_crop = detection['crop']
            score = detection['score']
            
            # Create filename: extract vehicle info from original filename
            base_name = os.path.splitext(vehicle_file)[0]
            plate_filename = f"{base_name}_plate{idx+1}_conf{score:.3f}.jpg"
            plate_path = output_dir / plate_filename
            
            # Save plate
            cv2.imwrite(str(plate_path), plate_crop)
            
            h, w = plate_crop.shape[:2]
            print(f"  ✓ Plate {idx+1}: {plate_filename} [{w}x{h}px] (Confidence: {score:.3f})")
            total_plates += 1
        
        print()
    
    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total vehicles processed: {len(vehicle_files)}")
    print(f"Vehicles with plates detected: {vehicles_with_plates}")
    print(f"Total plates saved: {total_plates}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")


def main():
    print("="*70)
    print("LICENSE PLATE DETECTION - Processing Detected Vehicles")
    print("="*70)
    
    # Check if directories exist
    if not VEHICLES_DIR.exists():
        print(f"\n❌ Vehicles directory not found: {VEHICLES_DIR}")
        print("   Please run vehicle detection first!")
        return
    
    if not MODEL_PATH.exists():
        print(f"\n❌ Plate model not found: {MODEL_PATH}")
        return
    
    # Load model
    print("\n[1/2] Loading plate detection model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device.upper()}")
    
    try:
        model, device = load_model(MODEL_PATH)
        print(f"  ✓ Model loaded successfully!")
    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return
    
    # Process vehicles
    print(f"\n[2/2] Detecting plates in vehicle images...")
    print(f"  Input: {VEHICLES_DIR}")
    print(f"  Output: {OUTPUT_PLATES_DIR}")
    print(f"  Confidence threshold: {SCORE_THRESHOLD}")
    
    try:
        process_vehicles(model, device, VEHICLES_DIR, OUTPUT_PLATES_DIR, SCORE_THRESHOLD)
        print("\n✓ Processing complete!")
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
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