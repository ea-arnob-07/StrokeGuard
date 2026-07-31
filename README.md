<div align="center">
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Smilies/Brain.png" alt="Brain" width="100" />
  
  <h1 align="center">StrokeGuard Clinical Decision Support System</h1>

  <p align="center">
    <strong>Turn scattered symptoms into a structured overview.</strong>
    <br />
    A guided symptom assessment experience created to organize health information and support better conversations with qualified medical professionals.
  </p>

  <p align="center">
    <a href="#"><img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
    <a href="#"><img src="https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"></a>
    <a href="#"><img src="https://img.shields.io/badge/Vercel-Deployed-black?style=for-the-badge&logo=vercel&logoColor=white" alt="Vercel"></a>
    <a href="#"><img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" alt="Status"></a>
  </p>
</div>

<hr />

## ✨ Overview

**StrokeGuard** is an advanced, beautifully designed clinical decision support interface that helps users assess stroke risks based on current health indicators and symptoms. It translates medical complexity into a clear, calm, and responsible assessment flow.

This application is strictly for **informational purposes** and is designed to complement—not replace—professional medical advice. 

---

## 🚀 Key Features

### 🧩 1. Structured Symptom Collection
Provides a comprehensive yet easy-to-understand form covering **17 structured inputs** including:
* **Cardiovascular Risks**: Chest pain, Hypertension, Irregular heartbeat, Shortness of breath
* **General Indicators**: Dizziness, Edema, Neck/Jaw pain, Excessive sweating
* **Other Markers**: Sleep apnea, Anxiety, Cold hands/feet, etc.

*(Fully supports bilingual interpretation: English & Bengali)*

### 📊 2. Doctor-Guided Result Presentation
Generates a portable, focused summary of the assessment. It provides:
* **Risk Categorization**: Clear visualization using dynamic gauges (Low, Elevated, Moderate, High, Severe Risk).
* **Pattern Overview**: A breakdown of the selected indicators.
* **Discussion Guide**: Useful topics and questions tailored for your next doctor's visit.

### 📄 3. Exportable PDF Reports
Generates a professional, medical-grade PDF report containing:
* Patient summary and selected symptoms.
* Graphical risk interpretation.
* Medical disclaimers and emergency awareness guidelines.

### 🎨 4. Premium Interface & Experience
* **Clarity First**: Modern, glass-morphic design with a responsive layout.
* **Context Over Certainty**: The estimate is always presented with its limitations.
* **Emergency Awareness**: Prominent warnings that sudden signs must never wait for an app result.

---

## 🏗️ Architecture & Technologies

StrokeGuard is built with modern, scalable technologies to ensure a robust user experience:

* **Frontend & Backend**: [Streamlit](https://streamlit.io/) (Python)
* **Machine Learning Engine**: Scikit-Learn, XGBoost, CatBoost, LightGBM
* **Visualizations**: Plotly (Interactive Gauges & Charts)
* **Report Generation**: ReportLab (PDFs)
* **UI/UX**: Custom CSS & HTML Injection for a premium, polished look.

---

## 🌐 Deployment (Vercel / Cloud)

This project is fully optimized for cloud deployment and runs smoothly on platforms like **Vercel** or **Streamlit Community Cloud**.

Since the application requires no database and handles everything statelessly via machine learning models and dynamic rendering, deployment is as simple as linking your repository to your hosting provider.

### Steps to Deploy on Vercel:
1. Connect your GitHub repository to Vercel.
2. Select **Python** as the framework (or leave it as standard if using `vercel.json`).
3. Set the build command to install `requirements.txt`.
4. Deploy! The app will automatically spin up the Streamlit server.

*(Note: No local installation or localhost configuration is required to use the deployed version.)*

---

## ⚠️ Medical Disclaimer

> **Professional medical guidance remains essential.**
> This application provides an experimental estimate based only on the information entered. Do not rely on the result alone for diagnosis, treatment, emergency decisions, fitness, or medical clearance. A qualified doctor's assessment and advice should always remain the primary basis for care.

---

<div align="center">
  <p>Product design & application engineering by <strong>Estiuk Arafat Arnob</strong></p>
  <p><em>Created with Clarity · Care · Craft</em></p>
</div>
