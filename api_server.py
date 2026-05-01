import numpy as np
from typing import Optional
import os
def to_python_type(obj):
    if isinstance(obj, dict):
        return {k: to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
import cv2
import base64
import time
import logging
import threading
import shutil
import uuid
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input as mobile_preprocess
from src.predictor import DogAgePredictor
import uvicorn
import sqlite3
import hashlib
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import models
import json
import re
from openai import OpenAI
import traceback
import tempfile
import requests

# BCS imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b3
from torchvision import transforms
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
predictor = DogAgePredictor(model_dir="models")

# OpenAI Setup for Skin Disease
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[WARNING] OPENAI_API_KEY is not set in .env")
client = OpenAI(api_key=OPENAI_API_KEY)

# Gemini Setup for BCS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY is not set in .env")
# Using REST API directly to avoid protobuf conflicts with TensorFlow

# Breed Identification Setup
BREED_MODEL = None
BREED_LABELS = None
DETECTOR_MODEL = MobileNetV2(weights="imagenet")
predict_lock = threading.Lock()

def load_breed_model():
    global BREED_MODEL, BREED_LABELS
    model_path = os.path.join("models", "dog_breed_model.h5")
    labels_path = os.path.join("models", "class_labels.json")
    if os.path.exists(model_path) and os.path.exists(labels_path):
        print(f"[API] Loading breed model from {model_path}...")
        BREED_MODEL = load_model(model_path)
        with open(labels_path, 'r') as f:
            BREED_LABELS = json.load(f)
        print("[API] Breed model and labels loaded successfully.")
    else:
        print(f"[WARNING] Breed model files not found at {model_path} or {labels_path}")

load_breed_model()

# Database setup
DB_PATH = "users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            dog_id TEXT NOT NULL,
            date TEXT NOT NULL,
            predicted_agerange TEXT,
            health_status TEXT,
            size_score REAL,
            weight REAL,
            image_data TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL,
            otp TEXT NOT NULL,
            expires_at REAL NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dogs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            breed TEXT,
            age INTEGER,
            age_range TEXT,
            weight REAL,
            last_skin_disease TEXT,
            last_emotion TEXT,
            created_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def seed_demo_data():
    """Seed sample dog profiles for demo and testing"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Create demo user if it doesn't exist
        cursor.execute("SELECT id FROM users WHERE email = ?", ("demo@dogtracker.com",))
        result = cursor.fetchone()
        
        if not result:
            hashed_pw = hashlib.sha256("demo123".encode()).hexdigest()
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                          ("Demo User", "demo@dogtracker.com", hashed_pw))
            demo_user_id = cursor.lastrowid
            conn.commit()
        else:
            demo_user_id = result[0]
        
        # Check if sample data already exists
        cursor.execute("SELECT COUNT(*) FROM dogs WHERE user_id = ?", (demo_user_id,))
        if cursor.fetchone()[0] == 0:
            sample_dogs = [
                ("Puppy", "Golden Retriever", 2, "0-1", 8.5, "Healthy", "Playful"),
                ("Max", "German Shepherd", 3, "1-5", 32.0, "Healthy", "Alert"),
                ("Bella", "Beagle", 2, "1-5", 13.5, "Healthy", "Excited"),
                ("Charlie", "French Bulldog", 4, "1-5", 12.0, "Healthy", "Calm"),
                ("Luna", "Siberian Husky", 7, "5-10", 28.5, "Healthy", "Friendly"),
                ("Rocky", "Boxer", 6, "5-10", 31.0, "Healthy", "Energetic"),
                ("Daisy", "Poodle", 12, "10+", 7.5, "Managed", "Calm"),
                ("Cooper", "Cocker Spaniel", 11, "10+", 14.5, "Healthy", "Gentle"),
                ("Buddy", "Labrador", 3, "0-1", 9.2, "Healthy", "Curious"),
            ]
            
            for name, breed, age, age_range, weight, health, emotion in sample_dogs:
                cursor.execute('''
                    INSERT INTO dogs (user_id, name, breed, age, age_range, weight, last_skin_disease, last_emotion, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (demo_user_id, name, breed, age, age_range, weight, health, emotion, "2026-03-12"))
            
            conn.commit()
            print("[DB] Sample dog profiles seeded successfully!")
            print("[DB] Demo user: demo@dogtracker.com / demo123")
    except Exception as e:
        print(f"[DB] Seeding error: {e}")
    finally:
        conn.close()

seed_demo_data()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

class UserSignup(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class UserProfileUpdate(BaseModel):
    name: str
    email: str

class UserRecord(BaseModel):
    user_id: int
    dog_id: str
    date: str
    predicted_agerange: Optional[str] = None
    health_status: Optional[str] = None
    size_score: Optional[float] = None
    weight: Optional[float] = None
    image_data: Optional[str] = None

class DogProfile(BaseModel):
    user_id: int
    name: str
    breed: Optional[str] = "Unknown"
    age: Optional[int] = 0
    age_range: Optional[str] = "Unknown"
    weight: Optional[float] = 0.0
    last_skin_disease: Optional[str] = None
    last_emotion: Optional[str] = None
    created_at: str

# BCS model setup
BCS_NUM_CLASSES = 3
BCS_CLASS_NAMES = ['Underweight', 'Healthy', 'Overweight']
BCS_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'bcs_model_final.pth')

def load_bcs_model():
    print("Loading BCS Model...")
    model = efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, BCS_NUM_CLASSES)
    model.load_state_dict(torch.load(BCS_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    print(f"[OK] Loaded BCS Model from {BCS_MODEL_PATH}")
    return model

bcs_model = load_bcs_model()

def preprocess_bcs_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(320),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

# Skin Disease model setup
SKIN_CLASSES = ['Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'None', 'demodicosis', 'ringworm']
SKIN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'skin_disease_model.pth')

def load_skin_model():
    print("Loading Skin Disease Model...")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(SKIN_CLASSES))
    checkpoint = torch.load(SKIN_MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("[OK] Loaded Skin Disease Model from", SKIN_MODEL_PATH)
    return model

skin_model = load_skin_model()

def preprocess_skin_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image

def get_disease_info_openai(disease_name):
    try:
        if disease_name.lower() == "none":
            prompt = """
            The image provided does not appear to be a dog's skin. 
            Provide a helpful explanation of why the detection might have failed (e.g., blur, poor lighting, or not being a dog).
            Respond ONLY in valid JSON format with the following structure:
            {
                "description": "A friendly explanation that the image doesn't appear to be dog skin.",
                "symptoms": "N/A or None (as a bulleted list with - at start of each line)",
                "causes": "Possible reasons for failed detection (as a bulleted list with - at start of each line)",
                "treatment": "Tips for taking a higher-quality photo of dog skin (as a bulleted list with - at start of each line)",
                "when_to_see_vet": "Advice to consult a vet if there is a genuine skin concern regardless of this result"
            }
            """
        elif disease_name.lower() == "healthy":
            prompt = """
            The dog's skin appears to be healthy. 
            Provide information about maintaining healthy dog skin and what healthy skin should look like.
            Respond ONLY in valid JSON format with the following structure:
            {
                "description": "A positive description of healthy dog skin.",
                "symptoms": "None (as a bulleted list with - at start of each line)",
                "causes": "N/A",
                "treatment": "General maintenance tips for keeping dog skin healthy (as a bulleted list with - at start of each line)",
                "when_to_see_vet": "When to perform routine checks or what signs to watch for"
            }
            """
        else:
            prompt = f"""
            Provide veterinary information about the dog skin disease "{disease_name}".
            Respond ONLY in valid JSON format with the following structure:
            {{
                "description": "A brief description of the disease",
                "symptoms": "Common symptoms (as a bulleted list with - at start of each line)",
                "causes": "Common causes (as a bulleted list with - at start of each line)",
                "treatment": "Typical treatment approaches (as a bulleted list with - at start of each line)",
                "when_to_see_vet": "Guidance on when to seek veterinary care"
            }}
            """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a veterinary expert. Respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        content = response.choices[0].message.content.strip()
        def parse_to_list(val):
            if isinstance(val, list):
                return val
            if not isinstance(val, str):
                return [str(val)]
            return [line.strip("- •").strip() for line in val.split('\n') if line.strip()]

        match = re.search(r'\{.*\}', content, re.DOTALL)
        data = {}
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(content)
            
        return {
            "description": data.get("description", ""),
            "symptoms": parse_to_list(data.get("symptoms", "")),
            "causes": parse_to_list(data.get("causes", "")),
            "treatment": parse_to_list(data.get("treatment", "")),
            "when_to_see_vet": data.get("when_to_see_vet", "")
        }
    except Exception as e:
        print(f"OpenAI error: {e}")
        return {
            "description": "Information currently unavailable.",
            "symptoms": ["Information not available"],
            "causes": ["Information not available"],
            "treatment": ["Please consult with a veterinarian"],
            "when_to_see_vet": "Consult a veterinarian if symptoms persist"
        }

def get_bcs_info_gemini(status):
    try:
        # Map "Healthy" to "Normal" for consistency with user preference
        display_status = "Normal" if status == "Healthy" else status
        
        # Using Google Gemini REST API directly to avoid Protobuf version conflicts with TensorFlow
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
        
        prompt = f"""
        A dog has been classified as: {display_status}
        (status can be: Underweight, Normal, Overweight)

        Based on this condition, provide:

        1. Diet recommendations (what to feed and what to avoid)
        2. Exercise plan (daily routine)
        3. Possible health risks (if any)
        4. Practical tips for the dog owner

        Guidelines:
        - Keep the answer simple and easy to understand
        - Use bullet points
        - Do not give medical prescriptions
        - Keep it within 120-150 words
        - Be practical and realistic

        Start with a short line:
        "Dog Condition: {display_status}"
        """
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        headers = {'Content-Type': 'application/json'}
        
        print(f"[Gemini] Requesting recommendations for: {display_status}...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 429:
            print("[Gemini] Error: Rate limit reached. You may be trying too fast for your API tier.")
            return "Suggestions are temporarily busy (Rate Limit). Please wait 1 minute and try again."
            
        response.raise_for_status()
        
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
            
        return "Recommendations currently unavailable. Please consult your veterinarian for specialized care advice."
    except Exception as e:
        print(f"Gemini REST Error: {e}")
        # Log more details if it's an HTTP error
        if hasattr(e, 'response') and e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response Body: {e.response.text}")
        elif "timeout" in str(e).lower():
            print("[Gemini] Error: The request timed out. Check your internet connection.")
            return "The request timed out. Please check your internet and try again."
        return "Recommendations currently unavailable. Please consult your veterinarian for specialized care advice."

# Allow CORS for Flutter frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Accepts either image or audio file, or both (for fusion)
from fastapi import Form
from typing import Optional, List
import os
import time
import logging

logging.basicConfig(level=logging.INFO)

@app.get("/")
async def root():
    return {"status": "online", "message": "Happypup API Backend is running successfully!"}

# Accepts either a single image (for upload mode) or multiple images (for camera mode)
@app.post("/predict")
async def predict(
    image: Optional[UploadFile] = File(None),
    images: Optional[List[UploadFile]] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    start_time = time.time()
    print("[API] Received prediction request.")
    logging.info("Received prediction request.")
    image_paths = []
    audio_path = None
    # Save single image if provided
    if image:
        image_path = f"temp_image_{image.filename}"
        with open(image_path, "wb") as f:
            f.write(await image.read())
        image_paths.append(image_path)
        print(f"[API] Saved image to {image_path}")
        logging.info(f"Saved image to {image_path}")
    # Save multiple images if provided
    if images:
        for idx, img in enumerate(images):
            img_path = f"temp_frame_{idx}_{img.filename}"
            with open(img_path, "wb") as f:
                f.write(await img.read())
            image_paths.append(img_path)
            print(f"[API] Saved frame to {img_path}")
    # Save audio if provided
    if audio:
        audio_path = f"temp_audio_{audio.filename}"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())
        print(f"[API] Saved audio to {audio_path}")
        logging.info(f"Saved audio to {audio_path}")
    save_time = time.time()
    print(f"[API] File save time: {save_time - start_time:.2f}s")
    logging.info(f"File save time: {save_time - start_time:.2f}s")
    # Run prediction
    pred_start = time.time()
    try:
        if len(image_paths) == 1:
            # Single image (upload or camera)
            results = predictor.predict_fusion(image_path=image_paths[0], audio_path=audio_path)
        elif len(image_paths) > 1:
            # Multiple frames: predict each, then majority vote
            preds = []
            for img_path in image_paths:
                res = predictor.predict_image(img_path)
                preds.append(res['age_group'])
            # Majority vote
            from collections import Counter
            most_common = Counter(preds).most_common(1)[0][0] if preds else None
            results = {"image": {"age_group": most_common, "all_frame_predictions": preds}}
        else:
            # No image, just audio
            results = predictor.predict_fusion(image_path=None, audio_path=audio_path)
    except Exception as e:
        print(f"[API] ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        results = {"error": str(e)}
    pred_end = time.time()
    print(f"[API] Prediction time: {pred_end - pred_start:.2f}s")
    logging.info(f"Prediction time: {pred_end - pred_start:.2f}s")
    # Clean up temp files
    for path in image_paths:
        if path and os.path.exists(path):
            os.remove(path)
    if audio_path and os.path.exists(audio_path):
        os.remove(audio_path)
    end_time = time.time()
    print(f"[API] Total request time: {end_time - start_time:.2f}s")
    logging.info(f"Total request time: {end_time - start_time:.2f}s")
    # Build a user-friendly response
    py_results = to_python_type(results)
    response = {}
    if py_results.get("image"):
        response["image_prediction"] = {
            "age_group": py_results["image"].get("age_group"),
            "confidence": py_results["image"].get("confidence"),
            "probabilities": py_results["image"].get("probabilities"),
            "all_frame_predictions": py_results["image"].get("all_frame_predictions")
        }
    if py_results.get("audio"):
        response["audio_prediction"] = {
            "age_group": py_results["audio"].get("age_group"),
            "confidence": py_results["audio"].get("confidence"),
            "probabilities": py_results["audio"].get("probabilities")
        }
    if py_results.get("fusion"):
        response["fusion_result"] = {
            "age_group": py_results["fusion"].get("age_group"),
            "confidence": py_results["fusion"].get("confidence"),
            "probabilities": py_results["fusion"].get("probabilities")
        }
    return response

# BCS prediction endpoint
from fastapi import Request
@app.post("/predict/bcs")
async def predict_bcs(image: UploadFile = File(...)):
    image_bytes = await image.read()
    try:
        input_tensor = preprocess_bcs_image(image_bytes)
        with torch.no_grad():
            outputs = bcs_model(input_tensor)
            probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            condition = BCS_CLASS_NAMES[pred_idx]
            confidence = int(probs[pred_idx] * 100)
            
        # Get AI recommendations using Gemini
        recommendations = get_bcs_info_gemini(condition)
            
        return {
            "condition": condition, 
            "confidence": confidence,
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/skin")
async def predict_skin(image: UploadFile = File(...)):
    try:
        image_bytes = await image.read()
        input_tensor = preprocess_skin_image(image_bytes)
        
        with torch.no_grad():
            outputs = skin_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred_class = probs.max(1)
        
        predicted_label = SKIN_CLASSES[pred_class.item()]
        confidence = float(conf.item() * 100)
        
        # Fetch disease info from OpenAI for all classes including "None" and "Healthy"
        disease_info = get_disease_info_openai(predicted_label)
        
        return {
            "predicted_disease": predicted_label.replace("_", " ").title(),
            "confidence": round(confidence, 2),
            "description": disease_info.get("description", ""),
            "symptoms": disease_info.get("symptoms", ""),
            "causes": disease_info.get("causes", ""),
            "treatment": disease_info.get("treatment", ""),
            "when_to_see_vet": disease_info.get("when_to_see_vet", ""),
            "status": "success"
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "message": str(e), "status": "error"}

# Authentication endpoints
@app.post("/signup")
async def signup(user: UserSignup):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        hashed_pw = hash_password(user.password)
        cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                       (user.name, user.email, hashed_pw))
        user_id = cursor.lastrowid
        conn.commit()
        return {"message": "User created successfully", "name": user.name, "user_id": user_id}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered")
    finally:
        conn.close()

class UserGoogleLogin(BaseModel):
    email: str
    name: str
    id: str
    photoUrl: Optional[str] = None

@app.post("/google-login")
async def google_login(user: UserGoogleLogin):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Check if user exists
        cursor.execute("SELECT id, name FROM users WHERE email = ?", (user.email,))
        result = cursor.fetchone()
        
        if result:
            user_id = result[0]
            # Update name if it changed
            cursor.execute("UPDATE users SET name = ? WHERE id = ?", (user.name, user_id))
            conn.commit()
            return {"message": "Login successful", "user_id": user_id, "name": user.name}
        else:
            # Create new user for Google login (password can be a random string or empty since it's not used)
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", 
                           (user.name, user.email, "GOOGLE_AUTH"))
            user_id = cursor.lastrowid
            conn.commit()
            return {"message": "User created and logged in", "user_id": user_id, "name": user.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.post("/login")
async def login(user: UserLogin):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    hashed_pw = hash_password(user.password)
    cursor.execute("SELECT id, name FROM users WHERE email = ? AND password = ?", 
                   (user.email, hashed_pw))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {"message": "Login successful", "user_id": result[0], "name": result[1]}
    else:
        raise HTTPException(status_code=401, detail="Invalid email or password")

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    email: str
    otp: str
    new_password: str

import random
import smtplib
from email.mime.text import MIMEText

# NOTE: You MUST change these credentials to use a real email sender!
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
# Use a real Gmail address here
SENDER_EMAIL = "your.email@gmail.com"
# Use a real App Password here (NOT your normal password)
SENDER_PASSWORD = "your_app_password_here"

def send_email_otp(to_email: str, otp: str):
    try:
        msg = MIMEText(f"Your Dog Tracker password reset code is: {otp}\n\nThis code will expire in 15 minutes.")
        msg["Subject"] = "Password Reset Code"
        msg["From"] = SENDER_EMAIL
        msg["To"] = to_email

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        # Only attempt login if credentials have been changed from defaults
        if SENDER_EMAIL != "your.email@gmail.com":
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
            print(f"[EMAIL] Successfully sent OTP to {to_email}")
        else:
            print("[EMAIL] WARNING: Email not sent! Default credentials used. OTP:", otp)
        server.quit()
        return True
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email: {e}")
        return False

@app.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id FROM users WHERE email = ?", (request.email,))
        if cursor.fetchone():
            # For testing purposes, we use a hardcoded OTP because no email sender is configured
            otp = "123456"
            # 15 minute expiration
            expires_at = time.time() + (15 * 60)
            
            # Store in DB
            cursor.execute("INSERT INTO password_resets (email, otp, expires_at) VALUES (?, ?, ?)", 
                           (request.email, otp, expires_at))
            conn.commit()
            
            # Send the email! (This will gracefully fail if credentials are default)
            send_email_otp(request.email, otp)
            
            print(f"\n{'='*50}")
            print(f"SECURITY ALERT: Test OTP generated for {request.email}")
            print(f"OTP CODE: {otp}")
            print(f"{'='*50}\n")
            
            return {"message": "Password reset link sent to your email"}
        
        # We return success even if not found to prevent email enumeration
        return {"message": "If that email exists, an OTP has been sent"}
    finally:
        conn.close()

@app.post("/reset-password")
async def reset_password(request: ResetPasswordRequest):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        current_time = time.time()
        # Find latest valid OTP
        cursor.execute(
            "SELECT id FROM password_resets WHERE email = ? AND otp = ? AND expires_at > ? ORDER BY id DESC LIMIT 1", 
            (request.email, request.otp, current_time)
        )
        reset_record = cursor.fetchone()
        
        if not reset_record:
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")
            
        # Valid OTP! Let's update the password
        hashed_pw = hash_password(request.new_password)
        cursor.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_pw, request.email))
        
        # Delete used OTP
        cursor.execute("DELETE FROM password_resets WHERE id = ?", (reset_record[0],))
        
        # Also clean up any other expired ones
        cursor.execute("DELETE FROM password_resets WHERE email = ? OR expires_at < ?", (request.email, current_time))
        
        conn.commit()
        return {"message": "Password reset successful"}
    finally:
        conn.close()

@app.post("/records")
async def save_record(record: UserRecord):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO user_records 
            (user_id, dog_id, date, predicted_agerange, health_status, size_score, weight, image_data) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (record.user_id, record.dog_id, record.date, record.predicted_agerange, 
              record.health_status, record.size_score, record.weight, record.image_data))
        conn.commit()
        return {"message": "Record saved successfully", "id": cursor.lastrowid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/records/{user_id}")
async def get_records(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_records WHERE user_id = ? ORDER BY id DESC", (user_id,))
    columns = [column[0] for column in cursor.description]
    results = []
    for row in cursor.fetchall():
        results.append(dict(zip(columns, row)))
    conn.close()
    return results

@app.get("/profile/{user_id}")
async def get_profile(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, email FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return {"name": result[0], "email": result[1]}
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.post("/dogs")
async def create_dog(dog: DogProfile):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO dogs (user_id, name, breed, age, age_range, weight, last_skin_disease, last_emotion, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (dog.user_id, dog.name, dog.breed, dog.age, dog.age_range, dog.weight, 
              dog.last_skin_disease, dog.last_emotion, dog.created_at))
        conn.commit()
        return {"message": "Dog created successfully", "id": cursor.lastrowid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/dogs/{user_id}")
async def get_dogs(user_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM dogs WHERE user_id = ?", (user_id,))
    columns = [column[0] for column in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

@app.put("/dogs/{dog_id}")
async def update_dog(dog_id: int, dog: DogProfile):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE dogs SET name=?, breed=?, age=?, age_range=?, weight=?, 
            last_skin_disease=?, last_emotion=? WHERE id=?
        ''', (dog.name, dog.breed, dog.age, dog.age_range, dog.weight, 
              dog.last_skin_disease, dog.last_emotion, dog_id))
        conn.commit()
        return {"message": "Dog updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.delete("/dogs/{dog_id}")
async def delete_dog(dog_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM dogs WHERE id = ?", (dog_id,))
        conn.commit()
        return {"message": "Dog deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.delete("/records/{record_id}")
async def delete_record(record_id: int):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM user_records WHERE id = ?", (record_id,))
        conn.commit()
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Record not found")
        return {"message": "Record deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def is_dog(img_path):
    """
    Uses MobileNetV2 to check if the image contains a dog.
    """
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = img_to_array(img_resized)
    img_preprocessed = mobile_preprocess(img_array)
    img_input = np.expand_dims(img_preprocessed, axis=0)

    preds = DETECTOR_MODEL.predict(img_input, verbose=0)
    decoded = decode_predictions(preds, top=10)[0] 

    dog_keywords = [
        'dog', 'puppy', 'terrier', 'hound', 'retriever', 'spaniel', 'collie', 
        'malamute', 'beagle', 'pug', 'mastiff', 'poodle', 'husky', 'dane', 
        'boxer', 'corgi', 'pointer', 'setter', 'sheepdog', 'chow', 'samoyed', 
        'spitz', 'rottweiler', 'pinscher', 'schnauzer', 'chihuahua', 'shih', 
        'pomeranian', 'basset', 'dalmatian', 'greyhound', 'bloodhound', 'dingo',
        'bulldog', 'whippet', 'weimaraner', 'newfoundland', 'saluki', 'dachshund',
        'kelpie', 'malinois', 'buhund', 'elkhound', 'affenpinscher', 'papillon',
        'cocker', 'pyrenees', 'borzoi', 'leonberg', 'kuvasz', 'schipperke', 'briard',
        'coyote', 'dhole', 'komondor', 'lhasa', 'vizsla', 'yorkie', 'afghan',
        'wolfhound', 'deerhound', 'foxhound', 'coonhound', 'pekinese', 'cairn', 
        'basenji', 'airedale', 'griffon', 'appenzeller', 'entlebucher'
    ]
    
    for _, label, score in decoded:
        label = label.lower()
        if any(key in label for key in dog_keywords) and score >= 0.02:
            return True
            
    return False

def preprocess_breed_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image {img_path} not found!")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_array = img_to_array(img_resized)
    img_preprocessed = mobile_preprocess(img_array)
    img_input = np.expand_dims(img_preprocessed, axis=0)

    return img_input, img_resized

def generate_grad_cam(model, img_array, class_idx):
    """
    Highly robust Grad-CAM that bypasses Sequential wrappers and handles nested models.
    """
    try:
        # 1. Find the base model and the target internal convolutional layer
        base_model = None
        target_layer = None
        
        # We look for the largest internal model or the model itself
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if isinstance(layer, tf.keras.Model) or (hasattr(layer, 'layers') and len(layer.layers) > 10):
                    base_model = layer
                    break
        
        if base_model is None:
            base_model = model

        def find_last_4d_layer(m):
            for layer in reversed(m.layers):
                if hasattr(layer, 'layers') and len(layer.layers) > 0:
                    res = find_last_4d_layer(layer)
                    if res: return res
                try:
                    if len(layer.output.shape) == 4:
                        return layer
                except:
                    continue
            return None

        target_layer = find_last_4d_layer(base_model)
        if not target_layer:
            print("[GRAD-CAM] Error: Could not find 4D layer.")
            return None
            
        print(f"[GRAD-CAM] Debug: Target='{target_layer.name}' in Base='{base_model.name}'")

        # 2. Build Gradient Model from the Base Model's input
        # This avoids "sequential never called" errors on the wrapper
        intermediate_model = tf.keras.models.Model(
            inputs=[base_model.input],
            outputs=[target_layer.output, base_model.output]
        )

        # 3. Compute gradients through the entire chain
        with tf.GradientTape() as tape:
            img_tensor = tf.cast(img_array, tf.float32)
            conv_outputs, base_preds = intermediate_model(img_tensor)
            
            # Map the base_preds through any layers AFTER base_model in the original model
            final_output = base_preds
            if base_model != model:
                found_base = False
                for layer in model.layers:
                    if found_base:
                        final_output = layer(final_output)
                    if layer == base_model:
                        found_base = True
            
            loss = final_output[:, class_idx]

        # Calculate gradients of the loss w.r.t the conv layer output
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            print("[GRAD-CAM] Error: Gradients are None.")
            return None

        # 4. Generate Heatmap
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ weights[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap).numpy()

        if max_val < 1e-9:
            # No strong signal found, return None instead of a misleading fallback
            return None

        heatmap = (heatmap / max_val).numpy()
        print(f"[GRAD-CAM] Success: Heatmap generated for {target_layer.name}")
        return heatmap

    except Exception as e:
        print(f"[GRAD-CAM] Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def overlay_heatmap(img_bgr, heatmap, alpha=0.5):
    """
    Overlay the heatmap on the original image.
    """
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    superimposed_img = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed_img

def tta_predict(img_path):
    """
    Test-Time Augmentation: run 5 variations of the image through the breed
    model and return the averaged softmax predictions.
    Variations: original, horizontal flip, brighter, darker, center-crop zoom.
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    variants = []

    # 1. Original
    variants.append(cv2.resize(img_rgb, (224, 224)))

    # 2. Horizontal flip
    variants.append(cv2.flip(cv2.resize(img_rgb, (224, 224)), 1))

    # 3. Brighter (+20%)
    variants.append(cv2.convertScaleAbs(cv2.resize(img_rgb, (224, 224)), alpha=1.2, beta=0))

    # 4. Darker (-20%)
    variants.append(cv2.convertScaleAbs(cv2.resize(img_rgb, (224, 224)), alpha=0.8, beta=0))

    # 5. Center crop — zoom in 15% to remove border noise
    mh, mw = int(h * 0.15), int(w * 0.15)
    cropped_center = img_rgb[mh:h - mh, mw:w - mw]
    variants.append(cv2.resize(cropped_center, (224, 224)))

    all_preds = []
    for v in variants:
        arr = img_to_array(v)
        arr = mobile_preprocess(arr)
        inp = np.expand_dims(arr, axis=0)
        pred = BREED_MODEL.predict(inp, verbose=0)[0]
        all_preds.append(pred)

    return np.mean(all_preds, axis=0)


def auto_crop_dog(img_path, initial_preds):
    """
    Use Grad-CAM on the breed model to locate the dog in the image and
    crop tightly around it. Returns the cropped image path, or None if
    cropping cannot improve things (box too small / grad-cam failed).
    """
    try:
        top_class = int(np.argmax(initial_preds))
        img_input, _ = preprocess_breed_image(img_path)
        heatmap = generate_grad_cam(BREED_MODEL, img_input, top_class)
        if heatmap is None:
            return None

        img_bgr = cv2.imread(img_path)
        h, w = img_bgr.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Threshold: keep the top-40% activation pixels
        threshold = np.percentile(heatmap_resized, 60)
        mask = (heatmap_resized >= threshold).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        all_pts = np.concatenate(contours)
        x, y, cw, ch = cv2.boundingRect(all_pts)

        # Add 10% padding on each side
        pad_x, pad_y = int(cw * 0.10), int(ch * 0.10)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + cw + pad_x)
        y2 = min(h, y + ch + pad_y)

        # Only crop if the detected region is at least 30% of the image
        # (avoids over-cropping when the dog fills the whole frame)
        if (x2 - x1) < w * 0.30 or (y2 - y1) < h * 0.30:
            print(f"[AUTO-CROP] Box too small ({x2-x1}×{y2-y1}), skipping crop.")
            return None

        # Don't bother if the crop is almost the whole image already
        if (x2 - x1) > w * 0.90 and (y2 - y1) > h * 0.90:
            print("[AUTO-CROP] Box covers full image, skipping crop.")
            return None

        cropped = img_bgr[y1:y2, x1:x2]
        crop_path = img_path.replace("temp_", "crop_")
        cv2.imwrite(crop_path, cropped)
        print(f"[AUTO-CROP] Cropped to ({x1},{y1})→({x2},{y2}) saved as {crop_path}")
        return crop_path

    except Exception as e:
        print(f"[AUTO-CROP] Error: {e}")
        return None


@app.post("/predict/breed")
async def predict_breed(file: UploadFile = File(...)):
    if BREED_MODEL is None or BREED_LABELS is None:
        raise HTTPException(status_code=500, detail="Breed model not loaded")
    
    unique_id = uuid.uuid4().hex
    temp_file_path = f"temp_{unique_id}_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        with predict_lock:
            if not is_dog(temp_file_path):
                return {
                    "breed": "Not a Dog",
                    "confidence": 0.0,
                    "mixed_breeds": "Please upload a clear picture of a dog.",
                    "grad_cam": None
                }

            # ── Step 1: TTA on the original image ──────────────────────────
            preds_array = tta_predict(temp_file_path)
            if preds_array is None:
                raise ValueError("TTA prediction failed")

            # ── Step 2: Auto-crop using Grad-CAM, then TTA on crop ─────────
            crop_path = auto_crop_dog(temp_file_path, preds_array)
            if crop_path and os.path.exists(crop_path):
                crop_preds = tta_predict(crop_path)
                if crop_preds is not None:
                    # Average original TTA with cropped TTA (weighted 40/60)
                    preds_array = 0.4 * preds_array + 0.6 * crop_preds
                    print("[BREED] Used auto-crop + TTA fusion")
                os.remove(crop_path)
            else:
                print("[BREED] Auto-crop skipped — using original TTA only")


        # --- Hybrid breed decoder ---
        # Maps known hybrid breed labels → their constituent pure breeds
        HYBRID_DECODER = {
            "Cockapoo":    ["Cocker", "Poodle"],
            "Labradoodle": ["Labrador", "Poodle"],
            "Cockapoo":    ["Cocker", "Poodle"],
        }

        # --- Get Top-10 breed predictions (more chances to find both mix breeds) ---
        top_n = min(10, len(preds_array))
        top_indices = np.argsort(preds_array)[-top_n:][::-1]

        top_breeds = [(BREED_LABELS[i], float(preds_array[i])) for i in top_indices]
        breed      = top_breeds[0][0]
        conf       = top_breeds[0][1]

        # --- Shannon Entropy ---
        all_probs  = np.clip(preds_array, 1e-9, 1.0)
        entropy    = float(-np.sum(all_probs * np.log(all_probs)))
        num_classes  = len(all_probs)
        max_entropy  = float(np.log(num_classes))
        norm_entropy = entropy / max_entropy

        # Confidence gap between 1st and 2nd breed
        conf_gap = conf - top_breeds[1][1]

        # --- If top-1 is a known hybrid, decode it to its constituents ---
        decoded_breeds = []
        if breed in HYBRID_DECODER:
            decoded_breeds = HYBRID_DECODER[breed]
            print(f"[BREED] Decoded hybrid '{breed}' → {decoded_breeds}")
        else:
            # Try to find constituent breeds in top-10
            # Collect all non-hybrid top-10 breeds as candidates
            decoded_breeds = [b for b, _ in top_breeds[:5] if b not in HYBRID_DECODER]

        # --- Normalize top-5 to get composition percentages ---
        top5_sum = sum(c for _, c in top_breeds[:5])
        if top5_sum > 0:
            composition = [(b, round(c / top5_sum * 100, 1)) for b, c in top_breeds[:5]]
        else:
            composition = [(b, 0.0) for b, _ in top_breeds[:5]]

        print(f"[BREED] top1={breed} conf={conf:.3f} gap={conf_gap:.3f} norm_entropy={norm_entropy:.3f}")
        print(f"[BREED] top-10: {[(b, round(c,3)) for b,c in top_breeds]}")

        # ── CALIBRATED THRESHOLDS (tuned on 19 labeled mixed breed images) ──
        PURE_ENTROPY_MAX = 0.15
        PURE_GAP_MIN     = 0.05

        if conf > 0.80 and conf_gap > PURE_GAP_MIN and norm_entropy < PURE_ENTROPY_MAX:
            # Very confident single breed → Purebred
            mixed_result = f"Purebred: {breed}"

        else:
            # Show top detected breeds — no percentages (they can be misleading for mixed breeds)
            detected = [b for b, p in composition if p > 5.0]
            if not detected:
                detected = [b for b, _ in composition[:2]]
            if len(detected) == 1:
                mixed_result = f"Detected Breed: {detected[0]}"
            else:
                mixed_result = "Detected Breeds: " + ",  ".join(detected)


        heatmap_base64 = None
        try:
            img_input, _ = preprocess_breed_image(temp_file_path)
            with predict_lock:
                heatmap = generate_grad_cam(BREED_MODEL, img_input, top_indices[0])

            if heatmap is not None:
                original_img = cv2.imread(temp_file_path)
                if original_img is not None:
                    superimposed = overlay_heatmap(original_img, heatmap, alpha=0.5)
                    _, buffer = cv2.imencode('.png', superimposed)
                    heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as grad_err:
            print(f"[GRAD-CAM] Execution Error: {grad_err}")
            traceback.print_exc()


        return {
            "breed": breed,
            "confidence": conf,
            "mixed_breeds": mixed_result,
            "grad_cam": heatmap_base64
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
