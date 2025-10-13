import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Detection AI - SWE1002 Project",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'pred_class' not in st.session_state:
    st.session_state.pred_class = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1E90FF, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .stage-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        margin: 0.2rem;
        border-radius: 25px;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, var(--color) 80%, #222 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        margin: 1rem 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid var(--accent-color);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #1E90FF, #FFD700);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(30, 144, 255, 0.3);
    }
    
    .upload-section {
        border: 2px dashed #1E90FF;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8faff 0%, #e8f4ff 100%);
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1E90FF;
        margin: 1rem 0;
    }
    
    .project-badge {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .developer-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        transition: transform 0.3s ease;
    }
    
    .developer-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
    }
    
    .course-info {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("alzimers_model.h5")

# Model and class definitions
model = load_model()
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
class_colors = {
    'MildDemented': '#FFD700',
    'ModerateDemented': '#FF8C00', 
    'NonDemented': '#32CD32',
    'VeryMildDemented': '#1E90FF'
}

class_descriptions = {
    'NonDemented': 'No signs of dementia detected. Brain function appears normal.',
    'VeryMildDemented': 'Very early stage with minimal cognitive decline.',
    'MildDemented': 'Mild cognitive impairment with noticeable symptoms.',
    'ModerateDemented': 'Moderate dementia with significant cognitive decline.'
}

def predict_image(img):
    """Predict Alzheimer's stage from MRI image"""
    img_resized = img.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    pred_idx = np.argmax(predictions)
    pred_class = class_names[pred_idx]
    confidence = float(np.max(predictions))
    
    return predictions[0], pred_class, confidence

def create_confidence_chart(predictions, class_names, colors):
    """Create a beautiful confidence chart"""
    df = pd.DataFrame({
        'Stage': [name.replace('Demented', ' Demented') for name in class_names],
        'Confidence': predictions * 100,
        'Color': [colors[name] for name in class_names]
    })
    
    fig = px.bar(
        df, 
        x='Confidence', 
        y='Stage',
        orientation='h',
        color='Color',
        color_discrete_map={color: color for color in df['Color']},
        title="Prediction Confidence by Stage"
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        font=dict(size=12),
        title_font_size=16,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(title="Confidence (%)", range=[0, 100])
    fig.update_yaxes(title="Alzheimer's Stage")
    
    return fig

def create_gauge_chart(confidence, pred_class, color):
    """Create a gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence Level<br><span style='font-size:0.8em;color:gray'>Predicted: {pred_class.replace('Demented', ' Demented')}</span>"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Main App Layout
def main():
    # Header with SWE1002 Project Badge
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="background: linear-gradient(45deg, #FF6B6B, #4ECDC4); padding: 0.5rem 1.5rem; border-radius: 25px; display: inline-block; margin-bottom: 1rem;">
            <h3 style="color: white; margin: 0; font-weight: bold;">🎓 SWE1002 Project</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🧠 Alzheimer\'s Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Deep Learning Model for MRI-based Alzheimer\'s Stage Detection</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # SWE1002 Project Info
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 15px; margin-bottom: 1rem;">
            <h3 style="color: white; text-align: center; margin: 0;">🎓 SWE1002 Project</h3>
            <p style="color: white; text-align: center; margin: 0.5rem 0; font-size: 0.9rem;">Software Engineering Course Project</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Meet Developers Section
        st.markdown("### 👥 Meet the Developers")
        developers = [
            {"name": "Naga Sri Nandhan Golla", "id": "24MIC7285", "icon": "👨‍💻"},
            {"name": "D N S Abhiram", "id": "24MIC7324", "icon": "👨‍💻"},
            {"name": "K Punith", "id": "24MIC7302", "icon": "👨‍💻"},
            {"name": "Siva Narayana Maila", "id": "24MIC7261", "icon": "👨‍💻"},
            {"name": "Lohitha", "id": "24MIC7274", "icon": "👩‍💻"}
        ]
        
        for dev in developers:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 0.8rem; border-radius: 12px; margin: 0.5rem 0;">
                <div style="color: white; font-weight: bold; font-size: 0.9rem;">
                    {dev['icon']} {dev['name']}
                </div>
                <div style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">
                    ID: {dev['id']}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📋 About This Tool")
        st.markdown("""
        This AI model analyzes MRI brain scans to detect and classify different stages of Alzheimer's disease using advanced deep learning techniques.
        """)
        
        st.markdown("### 🎯 Detection Stages")
        for stage, color in class_colors.items():
            stage_display = stage.replace('Demented', ' Demented')
            st.markdown(f"""
            <div style="background:{color}; color:white; padding:0.5rem; border-radius:10px; margin:0.2rem 0; text-align:center; font-weight:bold;">
                {stage_display}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### ℹ️ Instructions")
        st.markdown("""
        1. **Upload** an MRI brain scan image
        2. **Click Analyze** to process the image
        3. **View** detailed results and confidence metrics
        4. **Interpret** the prediction with medical guidance
        """)
        
        st.markdown("### ⚠️ Disclaimer")
        st.warning("This tool is for educational and research purposes only. Always consult healthcare professionals for medical diagnosis.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload MRI Image")
        
        # File uploader with custom styling
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear MRI brain scan image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Image", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("Analyzing MRI scan..."):
                    # Make prediction
                    predictions, pred_class, confidence = predict_image(image)
                    
                    # Store results in session state
                    st.session_state.predictions = predictions
                    st.session_state.pred_class = pred_class
                    st.session_state.confidence = confidence
                    st.session_state.analyzed = True
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if hasattr(st.session_state, 'analyzed') and st.session_state.analyzed:
            predictions = st.session_state.predictions
            pred_class = st.session_state.pred_class
            confidence = st.session_state.confidence
            color = class_colors[pred_class]
            
            # Main prediction card
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 80%, #222 100%); color: white; padding: 2rem; border-radius: 20px; text-align: center; margin: 1rem 0;">
                <h2 style="margin: 0; font-size: 2rem;">{pred_class.replace('Demented', ' Demented')}</h2>
                <p style="font-size: 1.2rem; margin: 0.5rem 0;">Confidence: {confidence:.1%}</p>
                <p style="font-size: 0.9rem; opacity: 0.9;">{class_descriptions[pred_class]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence gauge
            gauge_fig = create_gauge_chart(confidence, pred_class, color)
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        else:
            st.info("👆 Upload an MRI image and click 'Analyze' to see results")
    
    # Results section (full width)
    if hasattr(st.session_state, 'analyzed') and st.session_state.analyzed:
        st.markdown("---")
        st.markdown("### 📈 Detailed Analysis")
        
        col3, col4 = st.columns([2, 1])
        
        with col3:
            # Confidence chart
            conf_fig = create_confidence_chart(st.session_state.predictions, class_names, class_colors)
            st.plotly_chart(conf_fig, use_container_width=True)
        
        with col4:
            # Metrics
            st.markdown("#### 📋 Key Metrics")
            
            # Top prediction
            st.metric(
                label="Primary Prediction",
                value=st.session_state.pred_class.replace('Demented', ' Demented'),
                delta=f"{st.session_state.confidence:.1%} confidence"
            )
            
            # Secondary predictions
            sorted_indices = np.argsort(st.session_state.predictions)[::-1]
            
            st.markdown("#### 🔄 Alternative Predictions")
            for i, idx in enumerate(sorted_indices[1:3]):  # Show top 2 alternatives
                stage = class_names[idx].replace('Demented', ' Demented')
                conf = st.session_state.predictions[idx]
                st.metric(
                    label=f"#{i+2} Prediction",
                    value=stage,
                    delta=f"{conf:.1%}"
                )
        
        # Additional insights
        st.markdown("### 🧠 Clinical Insights")
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #ff6b9d; margin: 1rem 0; color: #333;">
                <h4 style="color: #d63384; margin-bottom: 1rem;">🎯 Prediction Reliability</h4>
                <p style="color: #495057; margin-bottom: 1rem;">High confidence predictions (>80%) are generally more reliable. Lower confidence may indicate:</p>
                <ul style="color: #495057;">
                    <li>Borderline cases between stages</li>
                    <li>Image quality issues</li>
                    <li>Need for additional clinical assessment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1.5rem; border-radius: 15px; border-left: 5px solid #20c997; margin: 1rem 0; color: #333;">
                <h4 style="color: #198754; margin-bottom: 1rem;">📋 Next Steps</h4>
                <p style="color: #495057; margin-bottom: 1rem;">Based on this analysis, consider:</p>
                <ul style="color: #495057;">
                    <li>Consulting with a neurologist</li>
                    <li>Additional cognitive assessments</li>
                    <li>Follow-up MRI scans if recommended</li>
                    <li>Lifestyle and treatment planning</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    # Project Information Section
    st.markdown("---")
    st.markdown("### 🎓 Project Information")
    
    proj_col1, proj_col2, proj_col3 = st.columns(3)
    
    with proj_col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; text-align: center;">
            <h4 style="color: white; margin: 0;">📚 Course</h4>
            <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">SWE1002</p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0;">Software Engineering</p>
        </div>
        """, unsafe_allow_html=True)
    
    with proj_col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 15px; text-align: center;">
            <h4 style="color: white; margin: 0;">🧠 Domain</h4>
            <p style="color: white; font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">Healthcare AI</p>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0;">Medical Image Analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with proj_col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 1.5rem; border-radius: 15px; text-align: center;">
            <h4 style="color: #333; margin: 0;">⚡ Technology</h4>
            <p style="color: #333; font-size: 1.2rem; font-weight: bold; margin: 0.5rem 0;">Deep Learning</p>
            <p style="color: #666; font-size: 0.9rem; margin: 0;">TensorFlow & Streamlit</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin: 1rem 0;">
        <h3 style="color: white; margin-bottom: 1rem;">🎓 SWE1002 - Alzheimer's Detection AI Project</h3>
        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin-bottom: 1rem;">
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: white;">
                👨‍💻 Naga Sri Nandhan Golla - 24MIC7285
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: white;">
                👨‍💻 D N S Abhiram - 24MIC7324
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: white;">
                👨‍💻 K Punith - 24MIC7302
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: white;">
                👨‍💻 Siva Narayana Maila - 24MIC7261
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; color: white;">
                👩‍💻 Lohitha - 24MIC7274
            </div>
        </div>
        <p style="color: rgba(255,255,255,0.9); margin: 0;">🔬 Powered by <strong>TensorFlow</strong> & <strong>Streamlit</strong></p>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0.5rem 0;">Built for educational and research purposes • Always consult healthcare professionals</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()