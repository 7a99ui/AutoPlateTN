"""
Batch Plate Enhancement Script
Reads all detected plates and applies ALL enhancement methods
Creates multiple enhanced versions for better OCR options
"""
import cv2
import numpy as np
from pathlib import Path
import os

# Get the base directory (where this script is located)
BASE_DIR = Path(__file__).parent.parent.parent.absolute()  # Go up to AutoPlateTN root


class PlateEnhancer:
    """Class for enhancing license plate image quality before OCR"""
    
    def __init__(self):
        """Initialize the enhancer"""
        pass
    
    def enhance(self, image, method='full'):
        """
        Enhance plate image quality
        
        Args:
            image: BGR numpy array or path to image
            method: 'full', 'light', 'aggressive', or 'none'
            
        Returns:
            Enhanced image (BGR format)
        """
        # Load image if it's a path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Cannot load image: {image}")
        else:
            img = image.copy()
        
        if img is None or img.size == 0:
            raise ValueError("Empty or invalid image")
        
        # Apply selected method
        if method == 'full':
            return self._full_enhancement(img)
        elif method == 'light':
            return self._light_enhancement(img)
        elif method == 'aggressive':
            return self._aggressive_enhancement(img)
        elif method == 'none':
            return img
        else:
            raise ValueError(f"Unknown method: {method}. Use 'full', 'light', 'aggressive', or 'none'")
    
    def _full_enhancement(self, image):
        """
        Full preprocessing (recommended for most cases)
        - Denoising
        - 4x Upscaling
        - CLAHE (contrast enhancement)
        - Gamma correction
        - Sharpening
        - Bilateral filter
        - Combined binarization
        - Morphology
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. Initial denoising
        gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. 4x upscaling to capture details
        h, w = gray.shape[:2]
        gray = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        
        # 3. CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 4. Gamma correction for better visibility
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        gray = cv2.LUT(gray, table)
        
        # 5. Sharpening to reduce blur
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # 6. Bilateral filter for smoothing while preserving edges
        sharpened = cv2.bilateralFilter(sharpened, d=5, sigmaColor=50, sigmaSpace=50)
        
        # 7. Polarity detection and inversion if needed
        mean_val = cv2.mean(sharpened)[0]
        if mean_val < 127:
            sharpened = cv2.bitwise_not(sharpened)
        
        # 8. Combined binarization (Otsu + Adaptive)
        _, binary_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_adaptive = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 31, 5
        )
        binary = cv2.bitwise_or(binary_otsu, binary_adaptive)
        
        # 9. Morphology to connect characters and clean
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_h, iterations=1)
        
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
        
        # Convert back to BGR for saving (3 channels)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def _light_enhancement(self, image):
        """
        Light preprocessing (for good quality images)
        - Light denoising
        - CLAHE
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Light denoising
        gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
        
        # Moderate CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Convert back to BGR
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def _aggressive_enhancement(self, image):
        """
        Aggressive preprocessing (for very degraded/blurry images)
        - Strong denoising
        - 6x upscaling
        - Aggressive CLAHE
        - Strong sharpening
        - Binarization
        - Strong morphology
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # 1. Strong denoising
        gray = cv2.fastNlMeansDenoising(gray, None, h=15, templateWindowSize=7, searchWindowSize=21)
        
        # 2. 6x upscaling
        h, w = gray.shape[:2]
        gray = cv2.resize(gray, (w * 6, h * 6), interpolation=cv2.INTER_CUBIC)
        
        # 3. Aggressive CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)
        
        # 4. Strong sharpening
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        # 5. Inversion if needed
        if cv2.mean(gray)[0] < 127:
            gray = cv2.bitwise_not(gray)
        
        # 6. Otsu binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 7. Strong morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Convert back to BGR
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def enhance_all_methods(self, image):
        """
        Apply all enhancement methods and return all results
        
        Args:
            image: Input image (BGR or path)
            
        Returns:
            dict with all enhancement results
        """
        # Load image if it's a path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()
        
        results = {
            'original': img,
            'light': self.enhance(img, method='light'),
            'full': self.enhance(img, method='full'),
            'aggressive': self.enhance(img, method='aggressive')
        }
        
        return results


def process_all_plates_all_methods(input_dir, output_base_dir):
    """
    Process all detected plates with ALL enhancement methods
    Creates separate folders for each method
    
    Args:
        input_dir: Directory containing detected plates
        output_base_dir: Base directory to save enhanced plates
    """
    # Create output directories for each method
    methods = ['original', 'light', 'full', 'aggressive']
    output_dirs = {}
    
    for method in methods:
        method_dir = output_base_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[method] = method_dir
    
    # Get all plate images
    plate_files = sorted([f for f in os.listdir(input_dir) 
                         if f.endswith(('.jpg', '.jpeg', '.png'))])
    
    if len(plate_files) == 0:
        print(f"❌ No plate images found in: {input_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Processing {len(plate_files)} plates with ALL enhancement methods")
    print(f"{'='*70}\n")
    
    # Initialize enhancer
    enhancer = PlateEnhancer()
    
    processed_count = 0
    failed_count = 0
    
    # Statistics
    stats = {method: 0 for method in methods}
    
    for plate_file in plate_files:
        input_path = input_dir / plate_file
        
        try:
            # Load image
            plate_img = cv2.imread(str(input_path))
            if plate_img is None:
                print(f"⚠️  Failed to load: {plate_file}")
                failed_count += 1
                continue
            
            h, w = plate_img.shape[:2]
            print(f"Processing: {plate_file} [{w}x{h}px]")
            
            # Apply all enhancement methods
            results = enhancer.enhance_all_methods(plate_img)
            
            # Save each version
            base_name = os.path.splitext(plate_file)[0]
            
            for method, enhanced_img in results.items():
                output_filename = f"{base_name}_{method}.jpg"
                output_path = output_dirs[method] / output_filename
                
                cv2.imwrite(str(output_path), enhanced_img)
                
                h_out, w_out = enhanced_img.shape[:2]
                print(f"  ✓ {method:12s}: {output_filename:50s} [{w_out}x{h_out}px]")
                stats[method] += 1
            
            processed_count += 1
            print()
            
        except Exception as e:
            print(f"❌ Error processing {plate_file}: {e}")
            failed_count += 1
            print()
    
    # Summary
    print(f"{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total plates: {len(plate_files)}")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"\nEnhancement Results:")
    for method in methods:
        print(f"  {method.capitalize():12s}: {stats[method]} images saved to {output_dirs[method]}")
    print(f"\nBase output directory: {output_base_dir}")
    print(f"{'='*70}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    print(f"  - Use 'full' method for most OCR tasks (best balance)")
    print(f"  - Use 'light' for high-quality plates")
    print(f"  - Use 'aggressive' for very blurry/low-quality plates")
    print(f"  - Compare results and choose the best for your OCR model")


def main():
    """Main function"""
    print("="*70)
    print("BATCH PLATE ENHANCEMENT - ALL METHODS")
    print("="*70)
    
    # Configuration (relative paths)
    INPUT_DIR = BASE_DIR / "data" / "processed" / "detected_plates"
    OUTPUT_BASE_DIR = BASE_DIR / "data" / "processed" / "enhanced_plates"
    
    # Check if input directory exists
    if not INPUT_DIR.exists():
        print(f"\n❌ Input directory not found: {INPUT_DIR}")
        print("   Please run plate detection first!")
        return
    
    print(f"\nConfiguration:")
    print(f"  Input:  {INPUT_DIR}")
    print(f"  Output: {OUTPUT_BASE_DIR}")
    print(f"  Methods: original, light, full, aggressive")
    print(f"\nOutput Structure:")
    print(f"  {OUTPUT_BASE_DIR}/")
    print(f"    ├── original/     (copy of detected plates)")
    print(f"    ├── light/        (light enhancement)")
    print(f"    ├── full/         (full enhancement - RECOMMENDED)")
    print(f"    └── aggressive/   (aggressive enhancement)")
    
    # Process all plates
    try:
        process_all_plates_all_methods(str(INPUT_DIR), str(OUTPUT_BASE_DIR))
        print("\n✓ Enhancement complete!")
        print(f"\n💡 Next step: Use enhanced plates for OCR")
        print(f"   Recommended: Use plates from 'full' folder for best results")
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