<!-- PROJECT STATUS BADGES -->
<p align="center">

  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" alt="status"/>
  <img src="https://img.shields.io/badge/Flask%20API-Ready-blue?style=for-the-badge" alt="flask api"/>
  <img src="https://img.shields.io/badge/Cloud%20Deployment-Pending-yellow?style=for-the-badge" alt="cloud deployment"/>
  <img src="https://img.shields.io/badge/React%20Native-Next%20Phase-orange?style=for-the-badge" alt="mobile app"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="license"/>

</p>

---

# 🍅 Tomato Disease Detection & Classification Dashboard

A web-based Machine Learning application that detects tomato leaf diseases using a trained deep learning model, built and deployed with *Streamlit Cloud*.

---

## 🌿 Overview

This project aims to help farmers and researchers quickly identify tomato plant diseases from leaf images using Artificial Intelligence.  
The app takes a tomato leaf image as input and classifies it into one of several disease categories — such as Early Blight, Late Blight, or Healthy Leaf — while also showing a confidence score.

---

## ⚙ Features

✅ Upload tomato leaf images directly through a clean Streamlit dashboard  
✅ Real-time AI-based disease prediction  
✅ Confidence visualization bar  
✅ Modern interface with background and custom footer  
✅ Fully deployed on Streamlit Cloud  

---

## 🚀 Run Locally

To run this project on your local system:

### 1. Clone the repository

```bash
git clone https://github.com/araomacaia/Tomato-Disease-Detection-App.git
cd Tomato-Disease-Detection-App
```
### 2. Create and activate a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate  # (Windows)
```

# OR

```bash
source .venv/bin/activate  # (Mac/Linux)
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🌐 Deployed Application
🔗 Live App: [Tomato Disease Detection Dashboard](https://tomato-disease-detection-app-d8iyuvfakt8sv7346ptfym.streamlit.app/)

---

## 🧠 Project Architecture & Future Development

This project has evolved beyond a simple dashboard into a **multi-layer AI system** combining local inference, cloud deployment, and mobile integration.

### 🧩 1. Model & Local Flask API
The trained TensorFlow model (`SavedModel` format) has been successfully integrated into a **Flask REST API**.  
This API:
- Accepts Base64-encoded tomato leaf images  
- Returns predicted disease and confidence  
- Supports both single and batch inference requests  

The Flask API runs smoothly in a local environment and is now being prepared for **cloud deployment**.

### ☁️ 2. Planned Cloud Deployment (Google Cloud Function)
The next step involves deploying the Flask API as a **Google Cloud Function**, allowing global accessibility through an HTTP endpoint.

Planned architecture:
- **Google Cloud Functions (GCF):** host the API serverless  
- **Google Cloud Storage:** store the trained TensorFlow model  
- **Google IAM:** manage access and permissions  

Once deployed, users and mobile apps will be able to send requests directly to the live API endpoint for real-time disease detection.

### 📱 3. React Native Mobile Integration
A **React Native mobile app** will be developed in the next phase to:
- Capture or upload leaf images  
- Send them to the cloud API  
- Display predictions and confidence levels in a user-friendly interface  

This integration will make the tomato disease detection system portable and accessible to farmers, researchers, and students globally.

---

## 🧾 Dataset

The model was trained using the **PlantVillage Dataset**, which contains images of healthy and diseased tomato leaves across multiple categories.

---

## 📈 Model Performance

| Metric | Value | 
|---------|-------|
| Training Accuracy | 92% |
| Validation Accuracy | 88% |
| Model Type | CNN (Convolutional Neural Network) |

---

## 🧪 Testing Scripts

Two Python testing scripts were created to validate the local Flask API:

| Script | Description |
|---------|-------------|
| `test_request.py` | Sends one tomato leaf image for prediction |
| `test_request_multiple.py` | Automatically loops through all images in the test folder and prints predictions for each |

Both have been verified successfully in the local environment.

---

## 🎯 Roadmap

- [x] Streamlit web app deployment  
- [x] Flask API local testing  
- [ ] Deploy Flask API to Google Cloud Functions  
- [ ] Build and connect React Native mobile app  
- [ ] Optimize model using TensorFlow Lite for faster inference  
- [ ] Publish technical documentation and user guide  

---

## Demo Video
🎥 Watch the demo: [Google Drive Link](https://drive.google.com/file/d/17uKnioe37-X6H7La6HSG924p9_Okoy-Y/view?usp=sharing)

---

## 👨‍💻 Developer

**Arao Zau Macaia**  
🎓 B.Tech in Electronics and Communication Engineering, NIT Durgapur  
📍 Elevate Labs AI & ML Intern  
📧 [araomacaia718@gmail.com](mailto:araomacaia718@gmail.com)  
🌐 [GitHub Profile](https://github.com/araomacaia)

---

## 🧾 License
This project is released under the MIT License — feel free to use and adapt it for research or educational purposes.

---

⭐ *If you like this project, consider giving it a star on GitHub!* ⭐

---

## 🧩 Citation (if included in research)
If this work is used in any academic or research context, please cite:
> Macaia, A. Z. (2025). *Tomato Disease Detection & Classification Dashboard*.  
> NIT Durgapur. GitHub Repository: [https://github.com/araomacaia/Tomato-Disease-Detection-App](https://github.com/araomacaia/Tomato-Disease-Detection-App)
