# 🏗️ StructLens - Civil Engineering Fault Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

<div align="center">
  <img src="https://img.icons8.com/fluency/96/000000/civil-engineering.png" width="120"/>
  <h3>Intelligent Structural Defect Detection & Analysis System</h3>
  <p>Harnessing Computer Vision & LLM for Civil Engineering Infrastructure Assessment</p>
</div>

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Usage Guide](#-usage-guide)
- [Model Training](#-model-training)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

**StructLens** is an advanced computer vision system designed for civil engineering applications that automatically detects, analyzes, and reports structural faults (cracks) in concrete infrastructure. The system combines traditional image processing techniques with machine learning and Large Language Models (LLM) to provide comprehensive engineering reports.

### Key Capabilities:
- ✅ Real-time crack detection from images
- ✅ Severity assessment (High/Medium/Low/None)
- ✅ ML-based classification (Random Forest)
- ✅ LLM-powered detailed engineering analysis
- ✅ Multi-format image support (JPG, PNG, WEBP, BMP, TIFF)
- ✅ Professional report generation

---

## ✨ Features

### 🖼️ Image Processing
- **Multi-format Support**: Upload images in JPG, PNG, WEBP, BMP, TIFF
- **Real-time Camera Input**: Capture photos directly
- **Sample Images**: Pre-loaded test images for demonstration
- **Image Preprocessing**: Grayscale conversion, thresholding, edge detection

### 🔬 Crack Analysis
- **Contour Detection**: Identifies crack patterns using OpenCV
- **Area Calculation**: Measures crack dimensions in pixels
- **Length Estimation**: Calculates crack perimeter/length
- **Severity Classification**: 
  - 🔴 **High**: Area > 1000 pixels
  - 🟠 **Medium**: Area > 200 pixels
  - 🟢 **Low**: Area ≤ 200 pixels
  - ⚪ **None**: No cracks detected

### 🤖 Machine Learning
- **Feature Extraction**: HOG, LBP, statistical features, Fourier transforms
- **Random Forest Classifier**: 100 estimators for robust prediction
- **Confidence Scoring**: Probability-based prediction confidence
- **Model Persistence**: Trained models saved as `.pkl` files

### 📝 LLM Integration
- **Local LLM Support**: Ollama integration (phi3:mini, llama2, mistral)
- **Free API Fallback**: Hugging Face Inference API
- **Engineering Reports**: Detailed analysis with:
  - Fault identification
  - Cause analysis
  - Prevention measures
  - Remediation recommendations

### 📊 Visualization & Reporting
- **Interactive Dashboard**: Streamlit-based UI
- **Multi-view Analysis**: 6-panel visualization grid
- **Download Options**: TXT reports and CSV summaries
- **Analysis History**: Track previous inspections

---

## 🏗️ System Architecture
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Input Image │────▶│ Preprocessing │────▶│ Crack Analysis │
│ (Upload/Camera)│ │ (Gray, Threshold)│ │ (Contours) │
└─────────────────┘ └──────────────────┘ └────────┬────────┘
│
▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ LLM Analysis │◀────│ ML Prediction │◀────│ Feature │
│ (Ollama/API) │ │(Random Forest) │ │ Extraction │
└────────┬────────┘ └──────────────────┘ └─────────────────┘
│
▼
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Engineering │────▶│ Visualization │────▶│ Report │
│ Report │ │ (6-panel plot) │ │ Generation │
└─────────────────┘ └──────────────────┘ └─────────────────┘


---

## 💻 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Web interface |
| **Image Processing** | OpenCV, PIL | Image manipulation |
| **Feature Extraction** | scikit-image, NumPy | HOG, LBP, FFT |
| **Machine Learning** | scikit-learn | Random Forest Classifier |
| **Deep Learning** | PyTorch (optional) | Model inference |
| **LLM Integration** | Ollama, Hugging Face | Text generation |
| **Data Handling** | Pandas, Joblib | Data management |
| **Visualization** | Matplotlib, Seaborn | Charts & plots |

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/structlens.git
cd structlens

# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows - Download from https://ollama.com/download
# Linux/Mac
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull phi3:mini
