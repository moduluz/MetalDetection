import os
import logging
import sys
import yaml
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, Union

class MetalDetector:
    """Metal detection system using YOLO model."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        """
        Initialize the metal detector.
        
        Args:
            model_path: Path to the trained YOLO model
            config_path: Optional path to configuration file
        """
        self._setup_logging()
        self.logger.info("Initializing Metal Detector...")
        
        # Validate and load model
        model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")
        self.model = YOLO(model_path)
        
        # Load configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Set default parameters
        self.conf_threshold = self.config.get('confidence_threshold', 0.25)
        self.iou_threshold = self.config.get('iou_threshold', 0.45)
        
        self.logger.info("Metal Detector initialized successfully")

    def _setup_logging(self) -> None:
        """Configure logging settings."""
        # Create logger
        self.logger = logging.getLogger('MetalDetector')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        try:
            # File handler
            file_handler = logging.FileHandler('metal_detector.log', encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"Warning: Could not set up logging: {str(e)}")
            # Fallback to basic logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger('MetalDetector')

    def predict_image(self, image_path: str, save_output: bool = True, 
                     display_result: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Perform metal detection on an image.
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated image
            display_result: Whether to display result
            
        Returns:
            Tuple of annotated image and detection results
        """
        self.logger.info(f"Processing image: {image_path}")
        
        # Validate input
        image_path = os.path.abspath(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")
            
        try:
            # Perform prediction
            results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold)
            
            # Process results
            detections = {}
            for result in results:
                # Get detection details
                boxes = result.boxes
                detections = {
                    'boxes': boxes.xyxy.cpu().numpy(),
                    'confidence': boxes.conf.cpu().numpy(),
                    'classes': boxes.cls.cpu().numpy()
                }
                
                # Annotate image
                annotated_img = result.plot()
                
                # Save output if requested
                if save_output:
                    output_dir = os.path.dirname(image_path)
                    output_path = os.path.join(output_dir, 
                                             Path(image_path).stem + "_detected.jpg")
                    cv2.imwrite(output_path, annotated_img)
                    self.logger.info(f"Saved annotated image: {output_path}")
                
                # Display if requested
                if display_result:
                    self._display_results(annotated_img, detections)
                
            return annotated_img, detections
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise

    def _display_results(self, img: np.ndarray, detections: Dict) -> None:
        """Display detection results with matplotlib."""
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Metal Detection Results')
        
        # Add detection information
        info_text = f"Detected Objects: {len(detections['boxes'])}\n"
        info_text += f"Average Confidence: {np.mean(detections['confidence']):.2f}"
        plt.text(10, 30, info_text, color='white', backgroundcolor='black')
        
        plt.show()

    def validate_model(self, data_yaml_path: str) -> Optional[Dict]:
        """
        Validate model performance.
        
        Args:
            data_yaml_path: Path to data.yaml file
            
        Returns:
            Dictionary containing validation metrics or None if validation fails
        """
        try:
            self.logger.info("Starting model validation...")
            
            # Check if data.yaml exists
            if not os.path.exists(data_yaml_path):
                self.logger.warning(f"Data file not found: {data_yaml_path}")
                return None
            
            
            # In the validate_model method
            results = self.model.val(data=data_yaml_path, batch=1)  # Process one image at a time
            
            # Extract metrics
            metrics = {
                'precision': float(results.boxes.precision.mean()),
                'recall': float(results.boxes.recall.mean()),
                'f1': float(results.boxes.f1.mean()),
                'mAP': float(results.maps.mAP)
            }
            
            # Log metrics
            self.logger.info("Validation metrics:")
            for metric, value in metrics.items():
                self.logger.info(f"{metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation error: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    try:
        # Initialize detector
        model_path = "C:/clone/metal detection/metal detection/runs/detect/train11/weights/best.pt"
        detector = MetalDetector(model_path=model_path)
        
        # Process single image
        image_path = "C:/clone/metal detection/metal detection/test.jpg"
        annotated_img, detections = detector.predict_image(image_path)
        
        # Optional: Validate model (may be skipped if dataset is not properly structured)
        data_yaml_path = "C:/clone/metal detection/metal detection/dataset/data.yaml"
        metrics = detector.validate_model(data_yaml_path)
        if metrics:
            print("\nValidation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")