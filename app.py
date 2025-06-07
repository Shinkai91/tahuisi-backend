import os
import re
import threading
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image
import vertexai
from vertexai.preview.generative_models import GenerativeModel

# Inisialisasi FastAPI
app = FastAPI()

# Memuat model TensorFlow yang telah disimpan
model = tf.saved_model.load('saved_model')

# Inisialisasi PaddleOCR untuk ekstraksi teks
ocr = PaddleOCR(use_angle_cls=True, lang='id')

# Daftar kandungan gizi yang ingin diekstrak
target_nutrients = ['Lemak Total', 'Protein', 'Gula', 'Garam']

# Fungsi untuk mengekstrak angka setelah label tertentu
def extract_number_after_label(text, label):
    text = text.lower().replace('(', ' ').replace(')', ' ')
    
    # Pola pencarian untuk setiap label
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
    
    # Mendapatkan pola pencarian untuk label tertentu
    label_patterns = patterns.get(label.lower())
    if not label_patterns:
        return 'N/A'
    
    # Membalikkan urutan pencarian pola untuk memulai dari sisi kanan
    for pattern in reversed(label_patterns):
        match = re.search(pattern, text)
        if match:
            value = match.group(1).replace(',', '.')
            if label.lower() == 'garam':
                return f"{value}mg"
            else:
                return f"{value}g"
    
    return 'N/A'

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    result_container = {}

    def process():
        contents = file.file.read()
        image = np.array(Image.open(BytesIO(contents)))
        input_tensor = tf.convert_to_tensor([image], dtype=tf.uint8)

        detections = model(input_tensor)
        boxes = detections['detection_boxes'].numpy()[0]
        scores = detections['detection_scores'].numpy()[0]

        h, w, _ = image.shape
        results = {key: 'N/A' for key in target_nutrients}

        confidence_threshold = 0.5
        valid_boxes = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score > confidence_threshold:
                valid_boxes.append(box)

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
                    for key in results.keys():
                        if results[key] == 'N/A':
                            value = extract_number_after_label(extracted_text, key)
                            if value != 'N/A':
                                results[key] = value

        service_account_path = "tahu-isi.json"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
        vertexai.init(project="tahu-isi", location="us-central1")

        nutrient_summary = ", ".join([f"{k}: {v}" for k, v in results.items()])
        prompt = f"""
        Berikut adalah informasi kandungan gizi dari suatu produk makanan:
        {nutrient_summary}.
        
        Tolong beri analisis singkat apakah kandungan ini termasuk sehat untuk dikonsumsi sehari-hari, dan berikan saran untuk konsumen jika ada.
        """

        generative_model = GenerativeModel("gemini-2.5-flash-preview-05-20")
        response = generative_model.generate_content(prompt)

        result_container["result"] = {
            "nutrients": results,
            "analysis": response.text
        }

    thread = threading.Thread(target=process)
    thread.start()
    thread.join()

    return JSONResponse(content=result_container["result"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)