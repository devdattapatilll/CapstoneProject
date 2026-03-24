# 📍 CivicTrack — AI-Powered Civic Issue Detection Platform

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![YOLOv12](https://img.shields.io/badge/YOLOv12-Object%20Detection-00FFFF?style=for-the-badge)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Firebase](https://img.shields.io/badge/Firebase-Auth%20%2B%20Firestore-FFCA28?style=for-the-badge&logo=firebase&logoColor=black)
![Render](https://img.shields.io/badge/Render-Backend-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Frontend-000000?style=for-the-badge&logo=vercel&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A smart community-driven platform to report civic issues — potholes, garbage, waterlogging, road cracks — with AI-powered automatic detection and classification.**

[🌐 Live Demo](https://capstone-project-vvlg.vercel.app) · [🔗 Backend API](https://civictrack-ml.onrender.com/health) · [📖 Deployment Guide](docs/DEPLOYMENT.md)

</div>

---

## 🎯 Overview

CivicTrack empowers citizens to report infrastructure issues in their community. When a user submits an issue with a photo and description:

1. **YOLOv12 vision models** scan the image for potholes, garbage, waterlogging, or road cracks
2. **NLP text classifier** auto-categorizes the complaint based on description
3. **Firebase** stores the issue with real-time status tracking
4. **Community Help Board** lets neighbours post urgent needs (blood donation, lost items, etc.)

---

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────────┐      ┌─────────────────┐
│   Vercel (CDN)  │      │   Render (Free Tier)  │      │    Firebase     │
│                 │      │                       │      │                 │
│  index.html     │─────▶│  Flask API (app.py)   │      │  Auth (Google)  │
│  styles.css     │      │                       │      │  Firestore DB   │
│  React 18 (CDN) │      │  ┌─────────────────┐  │      │  Cloud Storage  │
│                 │      │  │ 4 YOLO Models    │  │      │                 │
│  JS fetches ──────────▶│  │ pothole.pt (v12) │  │      │  issues         │
│  /analyze       │      │  │ garbage.pt       │  │      │  helpRequests   │
│  /classify-text │      │  │ waterlog.pt      │  │      │                 │
│                 │      │  │ crack.pt         │  │      │                 │
│  Firebase SDK ─────────────────────────────────────────▶│                 │
│                 │      │  ├─────────────────┤  │      │                 │
│                 │      │  │ NLP Classifier   │  │      │                 │
│                 │      │  │ classifier.pkl   │  │      │                 │
│                 │      │  └─────────────────┘  │      │                 │
└─────────────────┘      └──────────────────────┘      └─────────────────┘
```

---

## 📂 Project Structure

```
CapstoneProject/
│
├── app.py                    # Flask ML backend (REST API)
├── index.html                # Frontend — single-page React app
├── styles.css                # All CSS styles (responsive, dark-themed)
├── requirements.txt          # Python dependencies
├── Procfile                  # Render deployment start command
├── .gitignore                # Git exclusion rules
│
├── models/                   # Trained ML models (committed, ~24 MB total)
│   ├── pothole.pt            #   YOLOv12s — pothole detection
│   ├── garbage.pt            #   YOLOv8n  — garbage/waste detection
│   ├── waterlog.pt           #   YOLOv8n  — waterlogging/flood detection
│   ├── crack.pt              #   YOLOv8n  — road crack detection
│   └── classifier.pkl        #   TF-IDF + LinearSVC text classifier
│
├── training/                 # Model training scripts
│   ├── TRAIN_IN_COLAB.py     #   Google Colab guide (T4 GPU, 30 epochs)
│   ├── train_nlp.py          #   NLP classifier training (~200 samples)
│   └── train_yolo_local.py   #   Local CPU training (demo-quality)
│
└── docs/                     # Documentation
    └── DEPLOYMENT.md          #   Step-by-step deployment instructions
```

---

## 🤖 AI Models

### Computer Vision — YOLOv12 / YOLOv8

| Model | Base | Detects | Dataset Source |
|-------|------|---------|----------------|
| `pothole.pt` | **YOLOv12s** | Potholes, road damage | Roboflow: `pothole-jujbl` |
| `garbage.pt` | YOLOv8n | Waste, overflowing bins | Roboflow: `garbage-classification-3` |
| `waterlog.pt` | YOLOv8n | Flooding, waterlogging | Roboflow: `water-logging-aqhov` |
| `crack.pt` | YOLOv8n | Road cracks, surface damage | Roboflow: `crack-detection-yzwjm` |

### Text Classification — NLP

| Component | Details |
|-----------|---------|
| **Vectorizer** | TF-IDF (unigrams + bigrams, 5000 features) |
| **Classifier** | Calibrated LinearSVC |
| **Categories** | Roads, Garbage, Water Leakage, Electricity, Other |
| **Training Data** | ~200 hand-annotated civic complaint samples |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns model load status |
| `POST` | `/classify-text` | Classify text → category + confidence |
| `POST` | `/detect` | Run YOLO on image for a specific category |
| `POST` | `/detect-all` | Run all 4 YOLO models on one image |
| `POST` | `/analyze` | Full pipeline: text classification + image detection |
| `POST` | `/annotate` | Returns annotated image with bounding boxes |

### Example — Classify Text

```bash
curl -X POST https://civictrack-ml.onrender.com/classify-text \
  -H "Content-Type: application/json" \
  -d '{"text": "Huge pothole near school causing accidents"}'
```

Response:
```json
{
  "category": "Roads",
  "confidence": 0.86,
  "needs_review": false
}
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### 1. Clone & Install

```bash
git clone https://github.com/devdattapatilll/CapstoneProject.git
cd CapstoneProject
pip install -r requirements.txt
```

### 2. Train NLP Model (if not already trained)

```bash
python training/train_nlp.py
# Creates: models/classifier.pkl
```

### 3. Train YOLO Models (optional — pre-trained models included)

**Local (CPU, demo-quality):**
```bash
python training/train_yolo_local.py
```

**Google Colab (GPU, production-quality):**
- Upload `training/TRAIN_IN_COLAB.py` to [Google Colab](https://colab.research.google.com)
- Set runtime to **T4 GPU**
- Follow cell-by-cell instructions

### 4. Run Locally

```bash
python app.py
# Backend runs on http://localhost:5000
# Open index.html in browser (use VS Code Live Server)
```

---

## ☁️ Deployment

| Service | Purpose | URL | Plan |
|---------|---------|-----|------|
| **Render** | Flask ML backend | [civictrack-ml.onrender.com](https://civictrack-ml.onrender.com) | Free |
| **Vercel** | Static frontend | [capstone-project-vvlg.vercel.app](https://capstone-project-vvlg.vercel.app) | Free (Hobby) |
| **Firebase** | Auth + Database + Storage | — | Free (Spark) |

> ⚠️ **Note:** Render free tier sleeps after 15 min inactivity. First request after sleep takes ~50 seconds.

For step-by-step deployment instructions, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

---

## 🌟 Features

- **🔍 AI-Powered Detection** — YOLOv12 + NLP automatically detect and categorize issues
- **📸 Image Analysis** — Upload a photo and the AI finds potholes, garbage, cracks, flooding
- **📝 Smart Text Classification** — Description auto-categorized with confidence scores
- **📍 GPS Location** — One-click GPS location capture with Nominatim autocomplete
- **🔐 Firebase Auth** — Google Sign-In + Email/Password authentication
- **📊 Real-Time Dashboard** — Live issue stats (pending, in progress, resolved)
- **🤝 Community Help Board** — Blood donations, lost items, tutoring requests
- **👨‍💼 Admin Panel** — Update issue status, manage help requests
- **📱 Responsive Design** — Works on mobile, tablet, and desktop
- **🆓 100% Free** — All services use free tiers

---

## 💰 Cost Breakdown

| Service | Purpose | Cost |
|---------|---------|------|
| Firebase Auth | Login (Google + Email) | Free (Spark) |
| Firebase Firestore | Issue + Help Request storage | Free (1 GiB) |
| Firebase Storage | Uploaded images | Free (5 GB) |
| Render.com | Flask ML backend | Free (750 hrs/mo) |
| Vercel | Static frontend | Free (Hobby) |
| Roboflow | Training datasets | Free (Public) |
| Google Colab | YOLO training (T4 GPU) | Free |
| Nominatim | Location autocomplete | Free (OSM) |

**Total cost: ₹0**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | HTML5, CSS3, React 18 (CDN), Babel |
| **Backend** | Python, Flask, Flask-CORS, Gunicorn |
| **Computer Vision** | Ultralytics YOLOv12s, YOLOv8n |
| **NLP** | scikit-learn (TF-IDF + LinearSVC) |
| **Database** | Firebase Firestore (NoSQL) |
| **Auth** | Firebase Authentication |
| **Storage** | Firebase Cloud Storage |
| **Maps** | OpenStreetMap Nominatim API |
| **Hosting** | Render (backend), Vercel (frontend) |

---

## 👥 Team

| Name | Role | Responsibility |
|------|------|----------------|
| **Sarthak Sant** | Backend + UI/UX | Flask API, design system |
| **Mayank Rawat** | Authentication + Admin | Firebase Auth, admin panel |
| **Rizul Pathania** | Frontend | React components, responsive UI |
| **Devdatta Patil** | ML Models + Deployment | YOLO training, NLP, cloud deploy |

---

## 📝 License

This project is built for academic purposes as a Capstone Project.

---

<div align="center">

**Made with ❤️ for smarter communities**

[🌐 Try CivicTrack Live](https://capstone-project-vvlg.vercel.app)

</div>
