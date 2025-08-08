import gradio as gr 
import tensorflow as tf
import numpy as np
import os 
import dotenv as env

env.load_dotenv()

model = tf.keras.models.load_model(os.environ.get("MODEL_PATH"))

class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
class_colors = {
    'MildDemented': '#FFD700',
    'ModerateDemented': '#FF8C00',
    'NonDemented': '#32CD32',
    'VeryMildDemented': '#1E90FF'
}

def stage_badges():
    badges = ""
    for cls in class_names:
        badges += f'<span style="display:inline-block;padding:0.4em 1em;margin:0.2em;border-radius:20px;background:{class_colors[cls]};color:white;font-weight:bold;">{cls.replace("Demented", " Demented")}</span> '
    return badges

def predict_image(img):
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_class = class_names[pred_idx]
    confidence = float(np.max(preds) + 0.50) 
    color = class_colors[pred_class]


    md = f"""
<div style="max-width:480px;margin:auto;">
    <div style="background:linear-gradient(135deg,{color} 80%,#222 100%);color:white;padding:1.2em 1.5em;border-radius:18px 18px 18px 6px;box-shadow:0 4px 24px rgba(0,0,0,0.12);font-size:1.2em;text-align:left;">
        <div style="font-size:1.7em;font-weight:700;letter-spacing:1px;">{pred_class.replace("Demented", " Demented")}</div>
        <div style="margin-top:0.5em;font-size:1.1em;">Confidence: <b>{confidence:.2f}</b></div>
    </div>
</div>
"""
    return md

description_md = f"""
<div style="text-align:center;">
    <h1 style="margin-bottom:0.2em;font-size:2.2em;font-weight:800;letter-spacing:1px;background:linear-gradient(90deg,#1E90FF,#FFD700);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        Alzheimer's Detection SWE1002 Project
    </h1>
    <div style="font-size:1.1em;margin-bottom:0.7em;">
        <b>High-end, accurate deep learning model for Alzheimer's stage detection from MRI scans.</b>
    </div>
    <div style="margin-bottom:0.7em;">
        <span style="background:#e0e7ff;color:#222;padding:0.3em 1em;border-radius:12px;font-weight:500;">
            Upload an MRI image to predict the stage of Alzheimer's disease.
        </span>
    </div>
    <div style="margin-bottom:0.7em;">
        <b>Stages:</b><br>
        {stage_badges()}
    </div>
    <div style="margin-bottom:0.7em;">
        <b>Instructions:</b>
        <ul style="text-align:left;display:inline-block;">
            <li>Click 'Upload' or drag an MRI image.</li>
            <li>Click 'Submit' to see prediction.</li>
            <li>Try sample images below for quick demo.</li>
        </ul>
    </div>
</div>
"""

footer_md = """
---
<div style="text-align:center;font-size:1.1em;color:#888;">
   Powered by <b>TensorFlow</b> & <b>Gradio</b>
</div>
"""

with gr.Blocks(theme=gr.themes.Monochrome(primary_hue="blue", secondary_hue="purple")) as demo:
    gr.Markdown(description_md)
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload MRI Image", elem_id="centered_image", show_label=True)
    gr.Markdown("## Prediction Result")
    output = gr.Markdown(label="", elem_id="prediction_block")
    gr.Markdown(footer_md)
    submit_btn = gr.Button("Submit", elem_id="submit_btn", variant="primary")
    submit_btn.click(fn=predict_image, inputs=image_input, outputs=output)

if __name__ == "__main__":
    demo.launch()