#!/usr/bin/env python3
"""
Health check script for Streamlit Cloud deployment
"""

import os
import sys
import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        print("✅ TensorFlow version:", tf.__version__)
        print("✅ Streamlit version:", st.__version__)
        print("✅ NumPy version:", np.__version__)
        print("✅ Pandas version:", pd.__version__)
        print("✅ PIL available")
        return True
    except Exception as e:
        print("❌ Dependency check failed:", str(e))
        return False

def check_model():
    """Check if the model file exists and can be loaded"""
    try:
        if os.path.exists("alzimers_model.h5"):
            print("✅ Model file found")
            model = tf.keras.models.load_model("alzimers_model.h5")
            print("✅ Model loaded successfully")
            print("✅ Model input shape:", model.input_shape)
            print("✅ Model output shape:", model.output_shape)
            return True
        else:
            print("❌ Model file not found")
            return False
    except Exception as e:
        print("❌ Model loading failed:", str(e))
        return False

def main():
    """Run all health checks"""
    print("🔍 Running Streamlit Cloud deployment health checks...")
    print("=" * 50)
    
    deps_ok = check_dependencies()
    model_ok = check_model()
    
    print("=" * 50)
    if deps_ok and model_ok:
        print("🎉 All checks passed! Ready for Streamlit Cloud deployment.")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())