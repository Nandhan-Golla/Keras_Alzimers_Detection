---
title: Alzheimer's Detection AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# Alzheimer's Detection AI Model

This application uses a deep learning model to detect and classify different stages of Alzheimer's disease from MRI brain scans.

## Features

- **Real-time Analysis**: Upload MRI images for instant classification
- **Multi-stage Detection**: Identifies 4 different stages:
  - Non Demented (Healthy)
  - Very Mild Demented
  - Mild Demented  
  - Moderate Demented
- **Confidence Scoring**: Provides prediction confidence levels
- **User-friendly Interface**: Clean, intuitive Gradio interface

## How to Use

1. Upload an MRI brain scan image
2. Click "Submit" to analyze
3. View the predicted stage and confidence score

## Model Information

The model is trained using TensorFlow/Keras and processes 64x64 pixel MRI images to classify Alzheimer's progression stages.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or treatment.