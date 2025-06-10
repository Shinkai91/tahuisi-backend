# Nutrition Table OCR

Proyek ini adalah aplikasi backend berbasis FastAPI yang digunakan untuk memproses gambar tabel nutrisi dari produk makanan. Aplikasi ini menggunakan model TensorFlow untuk deteksi objek, PaddleOCR untuk ekstraksi teks, dan Vertex AI untuk menghasilkan analisis berdasarkan data nutrisi yang diekstrak.

## The Capstone Members

| Cohort ID  |         Nama         |          Email           |    Role     |
| :--------: | :------------------: | :----------------------: | :---------: |
| A671XBM145 |     Era Syafina      | A671XBM145@devacademy.id | AI Engineer |
| A208YAF202 | Hardianto Tandi Seno | A208YAF202@devacademy.id | AI Engineer |
| A010XBF387 |    Nurul Asyrifah    | A010XBF387@devacademy.id | AI Engineer |
| A200YBM515 |   Yosia Aser Camme   | A200YBM515@devacademy.id | AI Engineer |

## Tech Stack

Proyek ini dibangun menggunakan teknologi berikut:

<p align="center">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
    <img src="https://img.shields.io/badge/PaddleOCR-005BAC?style=for-the-badge&logo=paddlepaddle&logoColor=white" alt="PaddleOCR">
    <img src="https://img.shields.io/badge/Vertex%20AI-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Vertex AI">
    <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
    <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
    <img src="https://img.shields.io/badge/Uvicorn-FF69B4?style=for-the-badge&logo=uvicorn&logoColor=white" alt="Uvicorn">
</p>

## Fitur

- **Deteksi Objek**: Menggunakan TensorFlow untuk mendeteksi area tabel nutrisi pada gambar.
- **OCR**: Menggunakan PaddleOCR untuk mengekstrak teks dari tabel nutrisi.
- **Analisis Kandungan Gizi**: Menggunakan Vertex AI untuk memberikan analisis singkat tentang apakah kandungan gizi tersebut sehat untuk dikonsumsi sehari-hari.

## Cara Menggunakan di Lokal

### 1. Clone Repository

Clone repository ini ke komputer lokal Anda:

```bash
git clone https://github.com/Shinkai91/tahuisi-backend.git
cd tahuisi-backend
```

### 2. Install Dependencies

Pastikan Anda memiliki Python 3.10.0. Install dependencies menggunakan pip:

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

Jalankan aplikasi menggunakan Uvicorn:

```bash
python app.py
```

Aplikasi akan berjalan di `http://0.0.0.0:8080`.

### 4. Dokumentasi API

Buka dokumentasi API interaktif di `http://0.0.0.0:8080/docs` untuk mencoba endpoint yang tersedia.

### 5. Jalankan Aplikasi Menggunakan Docker

Pastikan Anda sudah menginstall Docker di komputer Anda. Ikuti langkah-langkah berikut untuk menjalankan aplikasi menggunakan Docker:

1. **Build Docker Image**  
   Jalankan perintah berikut untuk membangun image Docker:

   ```bash
   docker build -t tahuisi-backend .
   ```

2. **Jalankan Docker Container**  
   Setelah image berhasil dibuat, jalankan container menggunakan perintah berikut:

   ```bash
   docker run -d -p 8080:8080 tahuisi-backend
   ```

3. **Akses Aplikasi**  
   Aplikasi akan berjalan di `http://0.0.0.0:8080`. Anda juga dapat mengakses dokumentasi API di `http://0.0.0.0:8080/docs`.

## Deployment

Aplikasi ini dapat dideploy menggunakan platform cloud seperti Google Cloud Platform (GCP) atau layanan container seperti Docker dan Kubernetes. Pastikan untuk mengatur variabel lingkungan yang diperlukan sebelum deployment.

## Kontribusi

Made by **TahuIsi Team**  
Capstone Project **Laskar AI 2025**
