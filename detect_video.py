import torch
import torch.nn.functional as F
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import sys
from tqdm import tqdm
import mediapipe as mp  # Advanced face detection
import warnings
from scipy.stats import entropy
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
MODEL_PATH = "deepfake_model.pth"
CLASS_NAMES = ["Fake", "Real"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_FACE_CONFIDENCE = 0.7  # MediaPipe face detection confidence
DYNAMIC_FRAME_SAMPLING = True  # Auto-adjust frames based on video length
PLOT_ANALYSIS = True  # Generate visual analysis graphs

# ==================== ENHANCED FACE DETECTION ====================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=MIN_FACE_CONFIDENCE)

# ==================== PREPROCESSING (Match Training) ====================
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== MODEL LOADING ====================
def load_enhanced_model():
    """Load model with error handling and verification"""
    try:
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        
        # Verify model exists and is valid
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        
        # Quick verification forward pass
        test_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            model(test_input)
            
        return model
        
    except Exception as e:
        print(f"❌ Model loading failed: {str(e)}")
        sys.exit(1)

# ==================== ADVANCED FRAME PROCESSING ====================
def get_video_metadata(video_path):
    """Extract comprehensive video metadata"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
        
    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }
    cap.release()
    return metadata

def dynamic_frame_sampling(video_path):
    """Intelligently determine optimal frame count"""
    meta = get_video_metadata(video_path)
    if not meta:
        return 30  # Default fallback
        
    base_frames = 30
    duration = meta['duration']
    
    # Adjust frames based on video length
    if duration > 60:  # Long videos
        return min(100, int(duration * 0.5))  # 0.5 fps sampling
    elif duration > 30:
        return 50
    elif duration > 10:
        return 40
    else:
        return max(15, int(duration * 3))  # At least 3 fps for short clips

def detect_faces_mediapipe(frame):
    """State-of-the-art face detection with MediaPipe"""
    results = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = []
    
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            ih, iw = frame.shape[:2]
            x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
            w, h = int(bbox.width * iw), int(bbox.height * ih)
            
            # Expand bounding box by 15% for context
            expansion = 0.15
            x = max(0, int(x - w * expansion))
            y = max(0, int(y - h * expansion))
            w = min(iw - x, int(w * (1 + 2*expansion)))
            h = min(ih - y, int(h * (1 + 2*expansion)))
            
            faces.append((x, y, x+w, y+h, detection.score[0]))
    
    return sorted(faces, key=lambda x: x[4], reverse=True)  # Sort by confidence

# ==================== CORE DETECTION LOGIC ====================
def analyze_video_ultra(video_path):
    """Next-generation deepfake analysis pipeline"""
    # Load verified model
    model = load_enhanced_model()
    
    # Dynamic frame sampling
    num_frames = dynamic_frame_sampling(video_path) if DYNAMIC_FRAME_SAMPLING else 30
    print(f"🔍 Analyzing {num_frames} frames from video...")
    
    # Frame processing
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = max(total_frames // num_frames, 1)
    
    predictions = []
    frame_entropies = []
    processed_frames = []
    
    for i in tqdm(range(0, total_frames, frame_step), desc="Processing Frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Enhanced face detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces_mediapipe(frame_rgb)
        
        if not faces:
            print(f"⚠️ No face detected in frame {i}. Using full frame.")
            face_img = frame_rgb
        else:
            x1, y1, x2, y2, _ = faces[0]  # Use highest confidence face
            face_img = frame_rgb[y1:y2, x1:x2]
        
        pil_img = Image.fromarray(face_img)
        processed_frames.append((i, pil_img))  # Store for visualization
        
        # Calculate frame entropy (for artifact detection)
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        frame_entropies.append(entropy(hist, base=2))
        
        # Model prediction
        input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            predictions.append(probs.cpu().numpy()[0])
    
    cap.release()
    
    if not predictions:
        print("❌ No valid frames processed")
        return None
    
    # ==================== ADVANCED ANALYSIS ====================
    predictions = np.array(predictions)
    avg_probs = np.mean(predictions, axis=0)
    std_dev = np.std(predictions[:, 0])  # Fake score variability
    
    # Temporal consistency analysis
    rolling_avg = np.convolve(predictions[:, 0], np.ones(5)/5, mode='valid')
    temporal_inconsistency = np.std(rolling_avg)
    
    # Entropy analysis (higher entropy = more natural)
    avg_entropy = np.mean(frame_entropies)
    
    # ==================== RESULT INTERPRETATION ====================
    fake_prob = avg_probs[0]
    confidence = max(fake_prob, 1 - fake_prob)
    
    if fake_prob > 0.5:
        prediction = "Fake"
        confidence = fake_prob
    else:
        prediction = "Real"
        confidence = 1 - fake_prob
    
    # Generate comprehensive report
    report = {
        "prediction": prediction,
        "confidence": float(confidence),
        "fake_prob": float(fake_prob),
        "real_prob": float(avg_probs[1]),
        "temporal_inconsistency": float(temporal_inconsistency),
        "frame_entropy": float(avg_entropy),
        "processed_frames": len(predictions),
        "std_dev": float(std_dev)
    }
    
    # ==================== VISUALIZATION ====================
    if PLOT_ANALYSIS and len(predictions) > 10:
        plt.figure(figsize=(15, 5))
        
        # Probability plot
        plt.subplot(1, 2, 1)
        plt.plot(predictions[:, 0], label='Fake Probability')
        plt.axhline(y=0.5, color='r', linestyle='--')
        plt.title('Frame-by-Frame Prediction')
        plt.ylabel('Probability')
        plt.xlabel('Frame Index')
        plt.legend()
        
        # Entropy plot
        plt.subplot(1, 2, 2)
        plt.plot(frame_entropies, color='g')
        plt.title('Frame Entropy Analysis')
        plt.ylabel('Entropy (bits)')
        plt.xlabel('Frame Index')
        
        plt.tight_layout()
        plt.savefig('analysis_report.png')
        print("📊 Saved analysis report as 'analysis_report.png'")
    
    # ==================== VERDICT ====================
    print("\n=== 🔬 DEEPFAKE ANALYSIS REPORT ===")
    print(f"Final Verdict: {prediction} (Confidence: {confidence*100:.1f}%)")
    print(f"Average Fake Probability: {fake_prob*100:.1f}%")
    print(f"Temporal Inconsistency: {temporal_inconsistency:.3f}")
    print(f"Average Frame Entropy: {avg_entropy:.2f} bits")
    print(f"Standard Deviation: {std_dev*100:.1f}%")
    print(f"Frames Analyzed: {len(predictions)}")
    
    if temporal_inconsistency > 0.15:
        print("\n⚠️ Warning: High temporal inconsistency detected")
        print("This video shows signs of frame-by-frame manipulation")
    
    if avg_entropy < 6.5 and prediction == "Fake":
        print("\n🖼️ Visual artifacts detected:")
        print("Low entropy suggests compression artifacts or unnatural textures")
    
    return report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python detect_enhanced.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"Error: File not found - {video_path}")
        sys.exit(1)
    
    results = analyze_video_ultra(video_path)