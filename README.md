# 🐕 Dog Age Prediction & BMI Analysis

A multimodal deep learning system that predicts dog age (Young/Adult/Senior) from images and audio, then calculates breed-adjusted BMI.

## 📁 Project Structure

```
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── models/                 # Trained model weights
│   ├── best_balanced_image_model.pth
│   └── fast_audio_model_best.pth
├── data/
│   ├── images/            # Test images
│   └── audio/             # Test audio files (barks)
├── src/
│   ├── __init__.py
│   ├── models.py          # Neural network architectures
│   ├── predictor.py       # Age prediction with fusion
│   ├── audio_analysis.py  # Audio heuristics for better predictions
│   └── bmi_calculator.py  # BMI calculation with breed adjustments
└── dog_age_prediction.ipynb  # Training notebook (reference)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Predictions

```bash
# Image + Audio prediction
python main.py --image data/images/download.jpg --audio data/audio/puppy.aiff

# Image only
python main.py --image data/images/download.jpg

# Audio only
python main.py --audio data/audio/old.wav

# With BMI calculation
python main.py --image data/images/download.jpg --breed "labrador" --weight 25 --height 0.6

# Interactive mode (prompts for BMI inputs)
python main.py --image data/images/download.jpg --audio data/audio/bark.wav -i
```

### 3. List Supported Breeds

```bash
python main.py --list-breeds
```

## 🎯 How It Works

### Age Prediction

1. **Image Model**: EfficientNet-B0 fine-tuned on dog age dataset
2. **Audio Model**: CNN on mel-spectrograms + rule-based heuristics
3. **Fusion**: Confidence-weighted combination of both modalities

### Audio Heuristics

The audio model is enhanced with rule-based adjustments:

| Characteristic | Puppy Indicator | Senior Indicator |
|---------------|-----------------|------------------|
| Short bark (<1.5s) | ✅ +30% | ❌ -30% |
| High pitch | ✅ +25% | ❌ -25% |
| Energetic/fast tempo | ✅ +20% | ❌ -20% |
| Quiet/low RMS | ❌ -30% | ✅ +30% |
| Low energy | ❌ -20% | ✅ +20% |
| Has pauses | ❌ -15% | ✅ +15% |

Disable with `--no-heuristics` flag.

### BMI Calculation

BMI is adjusted for breed and age group:

```
Adjusted BMI = Base BMI × Breed Factor
Base BMI = Weight(kg) / Height(m)²
```

**Status Classification:**
- Underweight: < 70
- Normal: 70-90
- Overweight: > 90

## 📊 Example Output

```
==================================================
🐕 DOG AGE PREDICTION REPORT
==================================================

📷 IMAGE PREDICTION: Adult
   Confidence: 87.3%
   Probabilities: Young=5.2%, Adult=87.3%, Senior=7.5%

🔊 AUDIO PREDICTION: Adult
   Confidence: 72.1%
   Probabilities: Young=15.3%, Adult=72.1%, Senior=12.6%
   (Heuristics applied)
   Heuristic triggers: quiet

🎯 FUSION RESULT: Adult (adult)
   Confidence: 82.4%
   Fusion weights: Image=54.8%, Audio=45.2%

==================================================
📊 DOG BMI ANALYSIS REPORT
==================================================

🐕 Labrador Retriever (Adult)
   Weight: 25.0 kg
   Height: 0.6 m

📈 Base BMI: 69.44
   Adjustment factor: 1.0
   Adjusted BMI: 69.44

✅ Status: Underweight
   💡 Consider increasing food portions or consult a vet.
==================================================
```

## 🛠️ Advanced Usage

### Python API

```python
from src import DogAgePredictor, calculate_dog_bmi

# Initialize predictor
predictor = DogAgePredictor(model_dir="models")

# Predict age
results = predictor.predict_fusion(
    image_path="data/images/dog.jpg",
    audio_path="data/audio/bark.wav"
)

print(f"Predicted: {results['fusion']['prediction']}")
print(f"Age group: {results['fusion']['age_group']}")

# Calculate BMI
bmi = calculate_dog_bmi(
    weight_kg=25,
    height_m=0.6,
    breed="labrador",
    age_group=results['fusion']['age_group']
)
print(f"BMI Status: {bmi['status']}")
```

## 📝 Notes

- Models are already trained - no training required
- Image model works best; audio model enhanced with heuristics
- Supports common dog breeds (use `--list-breeds` to see all)
- Audio files: WAV, AIF, AIFF, MP3 formats supported
