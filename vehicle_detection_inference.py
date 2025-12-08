"""
Vehicle Detection Inference Script
Uses cloned detectron2 repository (no pip install needed)
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path

# Get the base directory (where this script is located)
BASE_DIR = Path(__file__).parent.absolute()

# Add detectron2 to path (relative to script location)
DETECTRON2_PATH = BASE_DIR / "detectron2"
if DETECTRON2_PATH.exists():
    sys.path.insert(0, str(DETECTRON2_PATH))
    print(f"✓ Added detectron2 from: {DETECTRON2_PATH}")
else:
    print(f"⚠️  Detectron2 not found at: {DETECTRON2_PATH}")
    print("   Please clone it first: git clone https://github.com/facebookresearch/detectron2.git")
    print("   Or ensure detectron2 folder is in the same directory as this script")

# Configuration (relative paths)
MODEL_PATH = BASE_DIR / "models" / "detection" / "cascade_best.pth"
INPUT_IMAGE = BASE_DIR / "data" / "samples" / "test1.jpg"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "detected_vehicules"
CLASS_NAMES = ["truck", "car", "van", "bus"]
CONFIDENCE_THRESHOLD = 0.5

def draw_boxes_opencv(image, boxes, classes, scores, class_names):
    """Draw bounding boxes using OpenCV"""
    img_copy = image.copy()
    colors = {
        0: (255, 100, 0),    # truck - blue
        1: (0, 255, 100),    # car - green
        2: (0, 200, 255),    # van - yellow
        3: (100, 0, 255)     # bus - red
    }
    
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(cls, (255, 255, 255))
        
        # Draw box with thicker line
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
        
        # Draw filled rectangle for label background
        label = f"{class_names[cls]}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw label background
        cv2.rectangle(img_copy, 
                     (x1, y1 - label_h - 10), 
                     (x1 + label_w + 10, y1),
                     color, -1)
        
        # Draw label text
        cv2.putText(img_copy, label, (x1 + 5, y1 - 5),
                   font, font_scale, (255, 255, 255), thickness)
    
    return img_copy

def load_model_and_predict(model_path, image, device):
    """Load model and run inference"""
    try:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
    except ImportError as e:
        print(f"\n❌ Failed to import detectron2: {e}")
        print("\nPlease ensure:")
        print("1. You have cloned detectron2: git clone https://github.com/facebookresearch/detectron2.git")
        print(f"2. Detectron2 folder is in: {BASE_DIR}")
        print(f"3. Current DETECTRON2_PATH: {DETECTRON2_PATH}")
        sys.exit(1)
    
    # Setup config
    cfg = get_cfg()
    
    # Use direct path to config file in the cloned repo
    config_file = DETECTRON2_PATH / "configs" / "Misc" / "cascade_mask_rcnn_R_50_FPN_3x.yaml"
    
    if not config_file.exists():
        # Fallback to faster rcnn config
        config_file = DETECTRON2_PATH / "configs" / "COCO-Detection" / "faster_rcnn_R_50_FPN_3x.yaml"
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        print(f"   Please ensure detectron2 configs exist at: {DETECTRON2_PATH / 'configs'}")
        sys.exit(1)
    
    print(f"  ✓ Using config: {config_file.name}")
    cfg.merge_from_file(str(config_file))
    
    # Manually set Cascade ROI Heads
    cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
    
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.DEVICE = device
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Run inference
    outputs = predictor(image)
    
    return outputs

def crop_and_save_detections(image, boxes, classes, scores, output_dir, class_names):
    """Crop detected vehicles and save them"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_vehicles = []
    for idx, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))
        y2 = max(y1+1, min(y2, h))
        
        # Crop vehicle
        vehicle_crop = image[y1:y2, x1:x2]
        
        # Skip if too small
        if vehicle_crop.size == 0 or vehicle_crop.shape[0] < 10 or vehicle_crop.shape[1] < 10:
            print(f"  ⚠️  Skipping vehicle {idx+1}: crop too small ({vehicle_crop.shape[1]}x{vehicle_crop.shape[0]}px)")
            continue
        
        # Save
        class_name = class_names[cls]
        filename = f"vehicle_{idx+1:03d}_{class_name}_{score:.3f}.jpg"
        output_path = output_dir / filename
        
        cv2.imwrite(str(output_path), vehicle_crop)
        
        vehicle_info = {
            'id': idx + 1,
            'class': class_name,
            'confidence': float(score),
            'bbox': [x1, y1, x2, y2],
            'crop_size': (vehicle_crop.shape[1], vehicle_crop.shape[0]),
            'saved_path': str(output_path)
        }
        saved_vehicles.append(vehicle_info)
        
        print(f"  ✓ {filename:40s} [{vehicle_crop.shape[1]:4d}x{vehicle_crop.shape[0]:4d}px]")
    
    return saved_vehicles

def main():
    print("="*70)
    print("VEHICLE DETECTION INFERENCE - Cascade R-CNN")
    print("="*70)
    
    # Verify files
    if not MODEL_PATH.exists():
        print(f"\n❌ Model not found: {MODEL_PATH}")
        return
    
    if not INPUT_IMAGE.exists():
        print(f"\n❌ Input image not found: {INPUT_IMAGE}")
        return
    
    # System info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'System Information':=^70}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {device.upper()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("="*70)
    
    # Load image
    print("\n[1/3] Loading image...")
    image = cv2.imread(str(INPUT_IMAGE))
    if image is None:
        print(f"❌ Failed to load: {INPUT_IMAGE}")
        return
    
    h, w = image.shape[:2]
    print(f"  ✓ Image size: {w}x{h} pixels")
    
    # Run detection
    print("\n[2/3] Running detection...")
    try:
        outputs = load_model_and_predict(MODEL_PATH, image, device)
        instances = outputs["instances"].to("cpu")
        num_detections = len(instances)
        print(f"  ✓ Detection complete! Found {num_detections} vehicles")
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    if num_detections == 0:
        print(f"\n⚠️  No vehicles detected (threshold: {CONFIDENCE_THRESHOLD})")
        print("   Try lowering the threshold in the script")
        return
    
    # Extract results
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    # Display results
    print(f"\n{'Detection Results':=^70}")
    print(f"{'ID':<4} {'Class':<10} {'Confidence':<12} {'Bounding Box':<25} {'Size'}")
    print("-"*70)
    for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = box
        w_box, h_box = x2-x1, y2-y1
        print(f"{i+1:<4} {CLASS_NAMES[cls]:<10} {score:.4f}       "
              f"[{int(x1):4d},{int(y1):4d},{int(x2):4d},{int(y2):4d}]  "
              f"{int(w_box):4d}x{int(h_box):4d}px")
    print("="*70)
    
    # Save crops
    print("\n[3/3] Saving detected vehicles...")
    saved_vehicles = crop_and_save_detections(
        image, boxes, classes, scores, OUTPUT_DIR, CLASS_NAMES
    )
    print(f"\n  ✓ Saved {len(saved_vehicles)}/{num_detections} vehicles")
    
    # Save visualization
    print("\nCreating visualization...")
    vis_image = draw_boxes_opencv(image, boxes, classes, scores, CLASS_NAMES)
    vis_path = OUTPUT_DIR / "detection_visualization.jpg"
    cv2.imwrite(str(vis_path), vis_image)
    print(f"  ✓ Saved: {vis_path.name}")
    
    # Summary
    print(f"\n{'Summary':=^70}")
    print(f"Input: {INPUT_IMAGE.name}")
    print(f"Detected: {num_detections} vehicles")
    print(f"Saved: {len(saved_vehicles)} crops")
    print(f"Output: {OUTPUT_DIR}")
    
    # Class distribution
    if saved_vehicles:
        class_counts = {}
        for v in saved_vehicles:
            class_counts[v['class']] = class_counts.get(v['class'], 0) + 1
        
        print(f"\nClass Distribution:")
        for cls in sorted(class_counts.keys()):
            count = class_counts[cls]
            bar = "█" * count
            print(f"  {cls.capitalize():<8}: {bar} ({count})")
    
    print("="*70)
    print("✓ Processing complete!")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()