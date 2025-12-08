"""
AutoPlateTN - Complete Pipeline Streamlit UI
Vehicle Detection → Plate Detection → Enhancement → OCR
"""

import streamlit as st
import cv2
import torch
import torchvision
import numpy as np
from pathlib import Path
import os
import sys
import tempfile
from PIL import Image
import pandas as pd
from torchvision.transforms import functional as F
from torchvision import transforms

# Page configuration
st.set_page_config(
    page_title="AutoPlateTN - License Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get the base directory (where app.py is located)
BASE_DIR = Path(__file__).parent.absolute()

# Add detectron2 to path
DETECTRON2_PATH = BASE_DIR / "detectron2"
if DETECTRON2_PATH.exists():
    sys.path.insert(0, str(DETECTRON2_PATH))

# Configuration paths (relative to BASE_DIR)
MODEL_PATHS = {
    'vehicle_detection': BASE_DIR / "models" / "detection" / "cascade_best.pth",
    'plate_detection': BASE_DIR / "models" / "detection" / "fasterrcnn_tunisia_plates.pth",
    'ocr_original': BASE_DIR / "models" / "ocr" / "best_crnn_model.pth",
    'ocr_light': BASE_DIR / "models" / "ocr" / "light" / "best_crnn_light.pth",
    'ocr_full': BASE_DIR / "models" / "ocr" / "full" / "best_crnn_full.pth",
    'ocr_aggressive': BASE_DIR / "models" / "ocr" / "aggressive" / "best_crnn_aggressive.pth"
}

CLASS_NAMES = ["truck", "car", "van", "bus"]
CHARACTERS = '0123456789T'
IMG_HEIGHT, IMG_WIDTH = 32, 128

# ==================== VEHICLE DETECTION ====================
class VehicleDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model_path = model_path
        self.predictor = None
        
    def load_model(self, confidence_threshold=0.5):
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            
            cfg = get_cfg()
            config_file = DETECTRON2_PATH / "configs" / "Misc" / "cascade_mask_rcnn_R_50_FPN_3x.yaml"
            
            if not config_file.exists():
                config_file = DETECTRON2_PATH / "configs" / "COCO-Detection" / "faster_rcnn_R_50_FPN_3x.yaml"
            
            cfg.merge_from_file(str(config_file))
            cfg.MODEL.ROI_HEADS.NAME = "CascadeROIHeads"
            cfg.MODEL.WEIGHTS = str(self.model_path)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
            cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
            cfg.MODEL.MASK_ON = False
            cfg.MODEL.DEVICE = self.device
            
            self.predictor = DefaultPredictor(cfg)
            return True
        except Exception as e:
            st.error(f"Failed to load vehicle detection model: {e}")
            return False
    
    def detect(self, image):
        if self.predictor is None:
            return None
        
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        
        return {
            'boxes': instances.pred_boxes.tensor.numpy(),
            'classes': instances.pred_classes.numpy(),
            'scores': instances.scores.numpy()
        }

# ==================== PLATE DETECTION ====================
class PlateDetector:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.model_path = model_path
        
    def load_model(self):
        try:
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes=2
            )
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.model = model
            return True
        except Exception as e:
            st.error(f"Failed to load plate detection model: {e}")
            return False
    
    def detect(self, image, score_threshold=0.5):
        if self.model is None:
            return []
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(img_rgb).to(self.device)
        
        with torch.no_grad():
            predictions = self.model([img_tensor])[0]
        
        boxes = predictions["boxes"].cpu()
        scores = predictions["scores"].cpu()
        
        detections = []
        for box, score in zip(boxes, scores):
            if score >= score_threshold:
                x1, y1, x2, y2 = box.int().tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'score': float(score),
                        'crop': image[y1:y2, x1:x2]
                    })
        
        return sorted(detections, key=lambda x: x['score'], reverse=True)

# ==================== PLATE ENHANCEMENT ====================
class PlateEnhancer:
    @staticmethod
    def enhance(image, method='full'):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        if method == 'light':
            gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        elif method == 'full':
            gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            gamma = 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            gray = cv2.LUT(gray, table)
            
            gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
            sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
            sharpened = cv2.bilateralFilter(sharpened, d=5, sigmaColor=50, sigmaSpace=50)
            
            if cv2.mean(sharpened)[0] < 127:
                sharpened = cv2.bitwise_not(sharpened)
            
            _, binary_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_adaptive = cv2.adaptiveThreshold(
                sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 31, 5
            )
            binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
            
            kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h, iterations=1)
            kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
            
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
        elif method == 'aggressive':
            gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (w * 6, h * 6), interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)
            
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            gray = cv2.filter2D(gray, -1, kernel)
            
            if cv2.mean(gray)[0] < 127:
                gray = cv2.bitwise_not(gray)
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        else:  # original
            return image

# ==================== OCR MODEL ====================
class CRNN(torch.nn.Module):
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
        return output.permute(1, 0, 2)

class PlateOCR:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = None
        self.characters = CHARACTERS
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            if 'characters' in checkpoint:
                self.characters = checkpoint['characters']
            
            num_classes = len(self.characters) + 1
            self.model = CRNN(IMG_HEIGHT, num_classes, hidden_size=256, num_layers=2)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(device)
            self.model.eval()
        except Exception as e:
            st.error(f"Failed to load OCR model: {e}")
    
    def recognize(self, image):
        if self.model is None:
            return {'raw': '', 'formatted': ''}
        
        # Preprocess
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img_tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
        
        # Decode
        _, pred = output.max(2)
        pred = pred.squeeze(1)
        
        chars = []
        prev_char = -1
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev_char:
                if idx - 1 < len(self.characters):
                    chars.append(self.characters[idx - 1])
            prev_char = idx
        
        raw_text = ''.join(chars)
        formatted_text = self.format_plate(raw_text)
        
        return {'raw': raw_text, 'formatted': formatted_text}
    
    def format_plate(self, raw_text):
        digits = ''.join(c for c in raw_text if c.isdigit())
        
        if 'T' in raw_text.upper():
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
            if len(digits) >= 6:
                return f"{digits[:6]} نت"
            else:
                return f"{digits} نت"
        else:
            if len(digits) >= 7:
                left = digits[:3]
                right = digits[3:]
                return f"{left} تونس {right}"
            elif len(digits) == 6:
                return f"{digits} نت"
            else:
                return raw_text

# ==================== UTILITY FUNCTIONS ====================
def draw_boxes(image, boxes, classes, scores):
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
        
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
        
        label = f"{CLASS_NAMES[cls]}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(img_copy, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(img_copy, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    
    return img_copy

# ==================== MAIN APP ====================
def main():
    st.title("🚗 AutoPlateTN - Automatic License Plate Recognition")
    st.markdown("**Complete Pipeline**: Vehicle Detection → Plate Detection → Enhancement → OCR")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"**Device**: {device.upper()}")
        if torch.cuda.is_available():
            st.success(f"GPU: {torch.cuda.get_device_name(0)}")
        
        st.divider()
        
        vehicle_conf = st.slider("Vehicle Detection Confidence", 0.0, 1.0, 0.5, 0.05)
        plate_conf = st.slider("Plate Detection Confidence", 0.0, 1.0, 0.5, 0.05)
        
        st.divider()
        
        enhancement_method = st.selectbox(
            "Enhancement Method",
            ['original', 'light', 'full', 'aggressive'],
            index=2
        )
        
        st.info(f"""
        **Enhancement Methods**:
        - **Original**: No enhancement
        - **Light**: Basic enhancement
        - **Full**: Recommended (4x upscale + CLAHE)
        - **Aggressive**: For very poor quality
        """)
        
        st.divider()
        
        use_matching_model = st.checkbox("Use matching OCR model", value=True)
        st.caption("If enabled, uses OCR model trained on the selected enhancement method")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.success(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Display original image
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        # Process button
        if st.button("🚀 Start Processing", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                # Step 1: Vehicle Detection
                st.markdown("---")
                st.subheader("1️⃣ Vehicle Detection")
                
                with st.spinner("Detecting vehicles..."):
                    vehicle_detector = VehicleDetector(MODEL_PATHS['vehicle_detection'], device)
                    if not vehicle_detector.load_model(vehicle_conf):
                        st.error("Failed to load vehicle detection model")
                        return
                    
                    vehicle_results = vehicle_detector.detect(image)
                    
                    if vehicle_results is None or len(vehicle_results['boxes']) == 0:
                        st.warning("No vehicles detected")
                        return
                    
                    num_vehicles = len(vehicle_results['boxes'])
                    st.success(f"✅ Detected {num_vehicles} vehicle(s)")
                    
                    # Display vehicles with boxes
                    vis_image = draw_boxes(
                        image, 
                        vehicle_results['boxes'], 
                        vehicle_results['classes'], 
                        vehicle_results['scores']
                    )
                    
                    with col2:
                        st.subheader("🚗 Detected Vehicles")
                        st.image(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                    
                    # Vehicle statistics
                    st.markdown("**Vehicle Statistics:**")
                    vehicle_df = pd.DataFrame({
                        'ID': range(1, num_vehicles + 1),
                        'Type': [CLASS_NAMES[c] for c in vehicle_results['classes']],
                        'Confidence': [f"{s:.3f}" for s in vehicle_results['scores']]
                    })
                    st.dataframe(vehicle_df, use_container_width=True)
                
                # Step 2: Plate Detection
                st.markdown("---")
                st.subheader("2️⃣ License Plate Detection")
                
                plate_detector = PlateDetector(MODEL_PATHS['plate_detection'], device)
                if not plate_detector.load_model():
                    st.error("Failed to load plate detection model")
                    return
                
                all_plates = []
                vehicle_crops = []
                
                for idx, (box, cls, score) in enumerate(zip(
                    vehicle_results['boxes'], 
                    vehicle_results['classes'], 
                    vehicle_results['scores']
                )):
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                    
                    vehicle_crop = image[y1:y2, x1:x2]
                    vehicle_crops.append(vehicle_crop)
                    
                    plates = plate_detector.detect(vehicle_crop, plate_conf)
                    
                    for plate in plates:
                        all_plates.append({
                            'vehicle_id': idx + 1,
                            'vehicle_type': CLASS_NAMES[cls],
                            'vehicle_conf': score,
                            'plate_crop': plate['crop'],
                            'plate_conf': plate['score']
                        })
                
                if len(all_plates) == 0:
                    st.warning("No license plates detected")
                    return
                
                st.success(f"✅ Detected {len(all_plates)} license plate(s)")
                
                # Display detected plates
                plate_cols = st.columns(min(len(all_plates), 4))
                for idx, plate_data in enumerate(all_plates):
                    with plate_cols[idx % 4]:
                        st.image(
                            cv2.cvtColor(plate_data['plate_crop'], cv2.COLOR_BGR2RGB),
                            caption=f"Vehicle {plate_data['vehicle_id']} - {plate_data['vehicle_type']}",
                            use_container_width=True
                        )
                
                # Step 3: Enhancement
                st.markdown("---")
                st.subheader("3️⃣ Plate Enhancement")
                
                st.info(f"**Method**: {enhancement_method.upper()}")
                
                enhancer = PlateEnhancer()
                
                enhanced_plates = []
                for plate_data in all_plates:
                    enhanced = enhancer.enhance(plate_data['plate_crop'], enhancement_method)
                    enhanced_plates.append(enhanced)
                    plate_data['enhanced_crop'] = enhanced
                
                st.success(f"✅ Enhanced {len(enhanced_plates)} plate(s)")
                
                # Display enhanced plates
                enh_cols = st.columns(min(len(enhanced_plates), 4))
                for idx, enhanced in enumerate(enhanced_plates):
                    with enh_cols[idx % 4]:
                        st.image(
                            cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB),
                            caption=f"Enhanced - Vehicle {all_plates[idx]['vehicle_id']}",
                            use_container_width=True
                        )
                
                # Step 4: OCR
                st.markdown("---")
                st.subheader("4️⃣ OCR - License Plate Recognition")
                
                # Select appropriate OCR model
                if use_matching_model:
                    if enhancement_method == 'original':
                        ocr_model_path = MODEL_PATHS['ocr_original']
                    elif enhancement_method == 'light':
                        ocr_model_path = MODEL_PATHS['ocr_light']
                    elif enhancement_method == 'full':
                        ocr_model_path = MODEL_PATHS['ocr_full']
                    elif enhancement_method == 'aggressive':
                        ocr_model_path = MODEL_PATHS['ocr_aggressive']
                else:
                    ocr_model_path = MODEL_PATHS['ocr_full']  # Default to full
                
                if not os.path.exists(ocr_model_path):
                    st.warning(f"OCR model not found: {ocr_model_path.name}. Using default model.")
                    ocr_model_path = MODEL_PATHS['ocr_original']
                
                st.info(f"**Using OCR Model**: {ocr_model_path.name}")
                
                ocr = PlateOCR(ocr_model_path, device)
                
                if ocr.model is None:
                    st.error("Failed to load OCR model")
                    return
                
                # Recognize plates
                results = []
                for idx, plate_data in enumerate(all_plates):
                    ocr_result = ocr.recognize(plate_data['enhanced_crop'])
                    results.append({
                        'Vehicle ID': plate_data['vehicle_id'],
                        'Vehicle Type': plate_data['vehicle_type'],
                        'Raw Text': ocr_result['raw'],
                        'Formatted': ocr_result['formatted'],
                        'Plate Confidence': f"{plate_data['plate_conf']:.3f}"
                    })
                
                st.success(f"✅ Recognized {len(results)} license plate(s)")
                
                # Display results
                st.markdown("### 📋 Recognition Results")
                
                results_df = pd.DataFrame(results)
                st.dataframe(results_df, use_container_width=True)
                
                # Display detailed results with images
                st.markdown("### 🔍 Detailed Results")
                
                for idx, (plate_data, result) in enumerate(zip(all_plates, results)):
                    with st.expander(f"Vehicle {result['Vehicle ID']} - {result['Vehicle Type']}", expanded=True):
                        result_cols = st.columns([1, 2])
                        
                        with result_cols[0]:
                            st.image(
                                cv2.cvtColor(plate_data['enhanced_crop'], cv2.COLOR_BGR2RGB),
                                caption="Enhanced Plate",
                                use_container_width=True
                            )
                        
                        with result_cols[1]:
                            st.markdown(f"""
                            **Raw Text:** `{result['Raw Text']}`
                            
                            **Formatted Plate:**
                            """)
                            st.markdown(f"<h2 style='text-align: center; color: #1f77b4;'>{result['Formatted']}</h2>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"**Confidence:** {result['Plate Confidence']}")
                
                # Summary
                st.markdown("---")
                st.subheader("📊 Processing Summary")
                
                summary_cols = st.columns(4)
                with summary_cols[0]:
                    st.metric("Vehicles Detected", num_vehicles)
                with summary_cols[1]:
                    st.metric("Plates Detected", len(all_plates))
                with summary_cols[2]:
                    st.metric("Plates Enhanced", len(enhanced_plates))
                with summary_cols[3]:
                    st.metric("Plates Recognized", len(results))
                
                st.success("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()