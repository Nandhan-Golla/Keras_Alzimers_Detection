import gradio as gr 
import tensorflow as tf
import numpy as np
import os 
import dotenv as env

env.load_dotenv()

model = tf.keras.models.load_model(os.environ.get("MODEL_PATH"))


class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))
    return f"Prediction: {pred_class} (Confidence: {confidence:.2f})"

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    title="Alzheimer's Detection",
    description="Upload an MRI image to predict Alzheimer's stage. (SWE1002 Project)"
    )

if __name__ == "__main__":
    iface.launch()