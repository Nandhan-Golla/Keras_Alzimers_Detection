# 🧠 Alzheimer's Detection AI - SWE1002 Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)

## 🎓 Academic Project Information
- **Course**: SWE1002 - Software Engineering
- **Domain**: Healthcare AI - Medical Image Analysis
- **Technology**: Deep Learning with TensorFlow & Streamlit

## 👥 Development Team
- 👨‍💻 **CHAMARTI KRISHNA MURTHY** - 24BCE7154
- 👨‍💻 **KOLIPAKULA LOKESH** - 24BCE8541
- 👨‍💻 **GANGULA THEERDHA SIVA NAGENDRA** - 24BCE8574
- 👨‍💻 **GEMBALI SRINIVAS RAO** - 24BCE8492
- 👨‍💻 **RUDRA PRATAP SINHA** - 24BCE7450



## 🚀 Application Overview

This advanced AI application uses deep learning to analyze MRI brain scans and detect different stages of Alzheimer's disease with high accuracy and beautiful visualizations.

### ✨ Key Features

- **🔍 Real-time Analysis**: Upload MRI images for instant AI-powered classification
- **📊 Advanced Visualizations**: Interactive charts, gauges, and confidence metrics
- **🎯 Multi-stage Detection**: Identifies 4 different Alzheimer's stages:
  - **Non Demented** (Healthy brain function)
  - **Very Mild Demented** (Early stage with minimal decline)
  - **Mild Demented** (Noticeable cognitive impairment)
  - **Moderate Demented** (Significant cognitive decline)
- **📈 Confidence Scoring**: Detailed prediction confidence with visual indicators
- **🎨 Stunning UI**: Professional interface with gradients and animations
- **🧠 Clinical Insights**: Medical guidance and next steps recommendations

### 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **Backend**: TensorFlow/Keras deep learning model
- **Visualization**: Plotly for interactive charts and gauges
- **Image Processing**: PIL for MRI image preprocessing
- **Data Analysis**: Pandas and NumPy for data manipulation

## 🚀 How to Use

1. **Upload**: Choose an MRI brain scan image (PNG, JPG, JPEG)
2. **Analyze**: Click the "🔍 Analyze Image" button
3. **Results**: View detailed predictions with:
   - Primary prediction with confidence level
   - Interactive confidence charts
   - Gauge visualization
   - Alternative predictions
   - Clinical insights and recommendations

## 🏥 Model Information

- **Architecture**: Convolutional Neural Network (CNN)
- **Input Size**: 64x64 pixel MRI images
- **Training**: Trained on medical MRI dataset
- **Accuracy**: High-performance classification model
- **Output**: 4-class classification with confidence scores

## 📋 Installation & Deployment

### Local Development
```bash
# Clone the repository
git clone <your-repo-url>
cd alzheimer-detection-ai

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Streamlit Cloud Deployment
1. Push your code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with `app.py` as the main file
4. Ensure `alzimers_model.h5` is included in the repository

## ⚠️ Important Disclaimer

**This application is developed for educational and research purposes as part of the SWE1002 Software Engineering course. It should NOT be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical advice.**

## 📄 License

This project is developed for academic purposes under the SWE1002 course curriculum.

---

**🎓 Developed by SWE1002 Students | 🔬 Powered by TensorFlow & Streamlit**