"""
Dog age prediction with multimodal fusion (image + audio).
"""
import os
import torch
import numpy as np
import librosa
from PIL import Image
from torchvision import transforms
from pathlib import Path

from .models import ImageModel, FastAudioModel
from .audio_analysis import analyze_audio, apply_audio_heuristics, get_audio_confidence_weight


# Constants
CLASSES = ['Young', 'Adult', 'Senior']
AGE_GROUP_MAP = {'Young': 'puppy', 'Adult': 'adult', 'Senior': 'senior'}

# Image preprocessing
TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class DogAgePredictor:
    """Multimodal dog age predictor using image and audio models."""
    
    def __init__(self, model_dir: str = None, device: str = None):
        """
        Initialize the predictor.
        
        Args:
            model_dir: Directory containing model files. Defaults to ../models/
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        if model_dir is None:
            # Default to models/ directory relative to project root
            model_dir = Path(__file__).parent.parent / "models"
        self.model_dir = Path(model_dir)
        
        self.image_model = None
        self.audio_model = None
        self._load_models()
    
    def _load_models(self):
        """Load pretrained models from disk."""
        # Load image model
        image_model_path = self.model_dir / "best_balanced_image_model.pth"
        if image_model_path.exists():
            self.image_model = ImageModel().to(self.device)
            self.image_model.load_state_dict(
                torch.load(str(image_model_path), map_location=self.device)
            )
            self.image_model.eval()
            print(f"[OK] Loaded image model from {image_model_path}")
        else:
            print(f"[WARN] Image model not found at {image_model_path}")
        
        # Load audio model
        audio_model_path = self.model_dir / "fast_audio_model_best.pth"
        if audio_model_path.exists():
            self.audio_model = FastAudioModel().to(self.device)
            self.audio_model.load_state_dict(
                torch.load(str(audio_model_path), map_location=self.device)
            )
            self.audio_model.eval()
            print(f"[OK] Loaded audio model from {audio_model_path}")
        else:
            print(f"[WARN] Audio model not found at {audio_model_path}")
    
    def predict_image(self, image_path: str) -> dict:
        """
        Predict age from image with Test-Time Augmentation (TTA).
        
        Returns:
        
            dict with 'prediction', 'confidence', 'probabilities'
        """
        if self.image_model is None:
            raise RuntimeError("Image model not loaded")
        
        img = Image.open(image_path).convert('RGB')
        
        # 1. Original Prediction
        img_tensor = TEST_TRANSFORM(img).unsqueeze(0).to(self.device)
        
        # 2. Flipped Prediction (TTA)
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tensor = TEST_TRANSFORM(flipped_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits_orig = self.image_model(img_tensor)
            probs_orig = torch.softmax(logits_orig, dim=1)[0].cpu().numpy()
            
            logits_flip = self.image_model(flipped_tensor)
            probs_flip = torch.softmax(logits_flip, dim=1)[0].cpu().numpy()
        
        # Average the probabilities (TTA)
        probs = (probs_orig + probs_flip) / 2.0
        
        pred_idx = int(np.argmax(probs))
        print(f"[PREDICTION] TTA applied (original vs flipped). Stability increased.")
        
        return {
            'prediction': CLASSES[pred_idx],
            'age_group': AGE_GROUP_MAP[CLASSES[pred_idx]],
            'confidence': float(probs[pred_idx]),
            'probabilities': {c: float(probs[i]) for i, c in enumerate(CLASSES)},
            'tta_info': {
                'original_confidence': float(probs_orig[np.argmax(probs_orig)]),
                'flipped_confidence': float(probs_flip[np.argmax(probs_flip)])
            }
        }
    
    def predict_audio(self, audio_path: str, apply_heuristics: bool = True) -> dict:
        """
        Predict age from audio with optional rule-based adjustments.
        
        Args:
            audio_path: Path to audio file
            apply_heuristics: Whether to apply audio analysis heuristics
            
        Returns:
            dict with 'prediction', 'confidence', 'probabilities', 'analysis'
        """
        if self.audio_model is None:
            raise RuntimeError("Audio model not loaded")
        
        # Load and preprocess audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Audio analysis for heuristics
        analysis = analyze_audio(y, sr)
        
        # Create mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Pad or truncate to fixed size
        if mel_db.shape[1] > 128:
            mel_db = mel_db[:, :128]
        else:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')
        
        spec_tensor = torch.tensor(
            mel_db[np.newaxis, np.newaxis, :], dtype=torch.float32
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.audio_model(spec_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        # Apply heuristics if enabled
        raw_probs = probs.copy()
        if apply_heuristics:
            probs = apply_audio_heuristics(probs, analysis)
        
        pred_idx = int(np.argmax(probs))
        confidence_weight = get_audio_confidence_weight(analysis)
        
        return {
            'prediction': CLASSES[pred_idx],
            'age_group': AGE_GROUP_MAP[CLASSES[pred_idx]],
            'confidence': float(probs[pred_idx]),
            'confidence_weight': confidence_weight,
            'probabilities': {c: float(probs[i]) for i, c in enumerate(CLASSES)},
            'raw_probabilities': {c: float(raw_probs[i]) for i, c in enumerate(CLASSES)},
            'analysis': analysis,
            'heuristics_applied': apply_heuristics
        }
    
    def predict_fusion(self, image_path: str = None, audio_path: str = None,
                       apply_audio_heuristics: bool = True) -> dict:
        """
        Multimodal prediction combining image and audio.
        
        Uses confidence-weighted fusion when both modalities are available.
        
        Args:
            image_path: Path to image file (optional)
            audio_path: Path to audio file (optional)
            apply_audio_heuristics: Whether to apply rule-based audio adjustments
            
        Returns:
            dict with fusion results
        """
        results = {
            'image': None,
            'audio': None,
            'fusion': None
        }
        
        probs_list = []
        weights = []
        
        # Image prediction
        if image_path and self.image_model:
            img_result = self.predict_image(image_path)
            results['image'] = img_result
            probs_list.append(np.array([img_result['probabilities'][c] for c in CLASSES]))
            weights.append(img_result['confidence'])
        
        # Audio prediction
        if audio_path and self.audio_model:
            aud_result = self.predict_audio(audio_path, apply_heuristics=apply_audio_heuristics)
            results['audio'] = aud_result
            probs_list.append(np.array([aud_result['probabilities'][c] for c in CLASSES]))
            # Use confidence weight from audio quality analysis
            weights.append(aud_result['confidence'] * aud_result['confidence_weight'])
        
        # Fusion
        if len(probs_list) == 2:
            weights = np.array(weights)
            
            # --- IMPROVED WEIGHTING ---
            # If one modality has very low confidence, rely entirely on the other
            # This prevents "random noise" from one model from ruining a confident prediction from the other
            CONF_THRESHOLD = 0.2
            img_conf = results['image']['confidence']
            aud_conf = results['audio']['confidence'] * results['audio']['confidence_weight']
            
            if img_conf > CONF_THRESHOLD and aud_conf <= CONF_THRESHOLD:
                weights = np.array([1.0, 0.0])
                print("[FUSION] Low audio confidence, relying on IMAGE.")
            elif aud_conf > CONF_THRESHOLD and img_conf <= CONF_THRESHOLD:
                weights = np.array([0.0, 1.0])
                print("[FUSION] Low image confidence, relying on AUDIO.")
            else:
                weights = weights / weights.sum()
            
            fused_probs = weights[0] * probs_list[0] + weights[1] * probs_list[1]
            pred_idx = int(np.argmax(fused_probs))
            
            results['fusion'] = {
                'prediction': CLASSES[pred_idx],
                'age_group': AGE_GROUP_MAP[CLASSES[pred_idx]],
                'confidence': float(fused_probs[pred_idx]),
                'probabilities': {c: float(fused_probs[i]) for i, c in enumerate(CLASSES)},
                'weights': {'image': float(weights[0]), 'audio': float(weights[1])}
            }
        elif len(probs_list) == 1:
            # Single modality - use as fusion result
            single_result = results['image'] or results['audio']
            results['fusion'] = {
                'prediction': single_result['prediction'],
                'age_group': single_result['age_group'],
                'confidence': single_result['confidence'],
                'probabilities': single_result['probabilities'],
                'weights': {'image': 1.0 if results['image'] else 0.0,
                           'audio': 1.0 if results['audio'] else 0.0}
            }
        
        return results


def print_prediction_report(results: dict):
    """Pretty print prediction results."""
    print("\n" + "="*50)
    print("🐕 DOG AGE PREDICTION REPORT")
    print("="*50)
    
    if results['image']:
        img = results['image']
        print(f"\n📷 IMAGE PREDICTION: {img['prediction']}")
        print(f"   Confidence: {img['confidence']:.1%}")
        print(f"   Probabilities: Young={img['probabilities']['Young']:.1%}, "
              f"Adult={img['probabilities']['Adult']:.1%}, "
              f"Senior={img['probabilities']['Senior']:.1%}")
    
    if results['audio']:
        aud = results['audio']
        print(f"\n🔊 AUDIO PREDICTION: {aud['prediction']}")
        print(f"   Confidence: {aud['confidence']:.1%}")
        print(f"   Probabilities: Young={aud['probabilities']['Young']:.1%}, "
              f"Adult={aud['probabilities']['Adult']:.1%}, "
              f"Senior={aud['probabilities']['Senior']:.1%}")
        if aud.get('heuristics_applied'):
            print(f"   (Heuristics applied - raw was: "
                  f"Young={aud['raw_probabilities']['Young']:.1%}, "
                  f"Adult={aud['raw_probabilities']['Adult']:.1%}, "
                  f"Senior={aud['raw_probabilities']['Senior']:.1%})")
            # Show which heuristics triggered
            analysis = aud['analysis']
            triggers = []
            # Puppy indicators
            if analysis.get('is_tonal'): triggers.append("tonal(puppy)")
            if analysis.get('is_high_tempo'): triggers.append("high_tempo(puppy)")
            if analysis.get('is_bursty'): triggers.append("bursty(puppy)")
            if analysis.get('is_short'): triggers.append("short(puppy)")
            # Adult indicators
            if analysis.get('has_moderate_silence'): triggers.append("moderate_silence(adult)")
            if analysis.get('is_high_zcr'): triggers.append("high_zcr(adult)")
            # Senior indicators
            if analysis.get('has_lots_of_silence'): triggers.append("lots_of_silence(senior)")
            if analysis.get('is_very_quiet') and analysis.get('is_low_energy'): 
                triggers.append("quiet+low_energy(senior)")
            if triggers:
                print(f"   Heuristic triggers: {', '.join(triggers)}")
    
    if results['fusion']:
        fus = results['fusion']
        print(f"\n🎯 FUSION RESULT: {fus['prediction']} ({fus['age_group']})")
        print(f"   Confidence: {fus['confidence']:.1%}")
        print(f"   Probabilities: Young={fus['probabilities']['Young']:.1%}, "
              f"Adult={fus['probabilities']['Adult']:.1%}, "
              f"Senior={fus['probabilities']['Senior']:.1%}")
        if 'weights' in fus:
            print(f"   Fusion weights: Image={fus['weights']['image']:.1%}, "
                  f"Audio={fus['weights']['audio']:.1%}")
    
    print("\n" + "="*50)
    return results['fusion']['age_group'] if results['fusion'] else None
