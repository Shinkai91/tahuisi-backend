import os
import re
from io import BytesIO
import logging

import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from paddleocr import PaddleOCR
from PIL import Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
ocr = None
generative_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model, ocr, generative_model
    
    try:
        logger.info("Loading TensorFlow model...")
        # Check if model directory exists
        if not os.path.exists('saved_model'):
            logger.error("saved_model directory not found")
            raise FileNotFoundError("saved_model directory not found")
        
        model = tf.saved_model.load('saved_model')
        logger.info("TensorFlow model loaded successfully")
        
        logger.info("Initializing PaddleOCR...")
        ocr = PaddleOCR(use_angle_cls=True, lang='id')
        logger.info("PaddleOCR initialized successfully")
        
        # Initialize Vertex AI
        logger.info("Initializing Vertex AI...")
        if not os.path.exists("tahu-isi.json"):
            logger.error("Google credentials file 'tahu-isi.json' not found")
            raise FileNotFoundError("Google credentials file not found")
        
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tahu-isi.json"
        vertexai.init(project="tahu-isi", location="us-central1")
        generative_model = GenerativeModel("gemini-2.5-flash-preview-05-20")
        logger.info("Vertex AI initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        # Don't raise the exception to allow the server to start
        # We'll handle missing models in the endpoint

@app.get("/")
async def root():
    return {"message": "Server is running!", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "tensorflow_model": model is not None,
        "paddleocr": ocr is not None,
        "vertex_ai": generative_model is not None
    }
    return {"status": "healthy", "models": status}

target_nutrients = ['Lemak Total', 'Protein', 'Gula', 'Garam']

def extract_number_after_label(text, label):
    """Extract nutrient values from OCR text"""
    try:
        text = text.lower().replace('(', ' ').replace(')', ' ')
        patterns = {
            'lemak total': [
                r'(\d+(?:[.,]\d+)?)\s*(?:g|gram)[^a-z]*(?:lemak\s+total|total\s+fat)',
                r'(?:lemak\s+total|total\s+fat)[^\d]*(\d+(?:[.,]\d+)?)\s*(?:g|gram)'
            ],
            'protein': [
                r'(\d+(?:[.,]\d+)?)\s*(?:g|gram)[^a-z]*protein',
                r'protein[^\d]*(\d+(?:[.,]\d+)?)\s*(?:g|gram)'
            ],
            'gula': [
                r'(\d+(?:[.,]\d+)?)\s*(?:g|gram)[^a-z]*(?:gula|sugar)',
                r'(?:gula|sugar)[^\d]*(\d+(?:[.,]\d+)?)\s*(?:g|gram)'
            ],
            'garam': [
                r'(\d+(?:[.,]\d+)?)\s*(?:mg|miligram)[^a-z]*(?:garam|natrium|sodium)',
                r'(?:garam|natrium|sodium)[^\d]*(\d+(?:[.,]\d+)?)\s*(?:mg|miligram)'
            ]
        }
        label_patterns = patterns.get(label.lower())
        if not label_patterns:
            return 'N/A'

        for pattern in reversed(label_patterns):
            match = re.search(pattern, text)
            if match:
                value = match.group(1).replace(',', '.')
                return f"{value}mg" if label.lower() == 'garam' else f"{value}g"
        return 'N/A'
    except Exception as e:
        logger.error(f"Error extracting label {label}: {str(e)}")
        return 'N/A'

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    """Process uploaded image for nutrient extraction"""
    try:
        # Check if models are loaded
        if model is None:
            raise HTTPException(status_code=503, detail="TensorFlow model not loaded")
        if ocr is None:
            raise HTTPException(status_code=503, detail="PaddleOCR not initialized")
        if generative_model is None:
            raise HTTPException(status_code=503, detail="Vertex AI not initialized")
        
        # Validate file
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        logger.info(f"Processing image: {file.filename}")
        
        # Read and process image
        contents = await file.read()
        image = np.array(Image.open(BytesIO(contents)))
        
        # TensorFlow model inference
        input_tensor = tf.convert_to_tensor([image], dtype=tf.uint8)
        detections = model(input_tensor)
        
        boxes = detections['detection_boxes'].numpy()[0]
        scores = detections['detection_scores'].numpy()[0]
        
        h, w, _ = image.shape
        results = {key: 'N/A' for key in target_nutrients}

        confidence_threshold = 0.5
        valid_boxes = [box for i, box in enumerate(boxes) if scores[i] > confidence_threshold]

        if valid_boxes:
            box = valid_boxes[0]
            ymin, xmin, ymax, xmax = box
            xmin, xmax, ymin, ymax = map(int, [xmin * w, xmax * w, ymin * h, ymax * h])
            cropped = image[ymin:ymax, xmin:xmax]
            
            if cropped.size > 0:
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                ocr_result = ocr.ocr(gray, cls=True)
                
                if ocr_result and ocr_result[0]:
                    extracted_text = ' '.join([line[1][0] for line in ocr_result[0]])
                    logger.info(f"Extracted text: {extracted_text}")
                    
                    for key in results:
                        value = extract_number_after_label(extracted_text, key)
                        if value != 'N/A':
                            results[key] = value

        # Generate analysis using Vertex AI
        prompt = (
            "Berikut adalah informasi kandungan gizi dari suatu produk makanan:\n"
            f"{', '.join(f'{k}: {v}' for k, v in results.items())}.\n\n"
            "Tolong beri analisis singkat apakah kandungan ini termasuk sehat untuk dikonsumsi sehari-hari, dan berikan saran untuk konsumen jika ada."
        )

        response = generative_model.generate_content(prompt)

        return JSONResponse(content={
            "nutrients": results,
            "analysis": response.text
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    # Get port from environment variable or default to 8080
    port = int(os.environ.get("PORT", 8080))
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )