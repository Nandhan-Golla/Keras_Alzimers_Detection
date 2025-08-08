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

def predict_image(img):
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    pred_class = class_names[pred_idx]
    confidence = float(np.max(preds))
    color = class_colors[pred_class]
    # Markdown output with color
    md = f"""
<div style="padding:1em;border-radius:10px;background:{color};color:white;text-align:center;font-size:1.3em;">
<b>Prediction:</b> {pred_class}<br>
<b>Confidence:</b> {confidence:.2f}
</div>
"""
    return md

examples = [
    ["examples/mild.jpg"],
    ["examples/moderate.jpg"],
    ["examples/non.jpg"],
    ["examples/verymild.jpg"]
]

description_md = """
# 🧠 Alzheimer's Detection (SWE1002 Project)

Upload an MRI image to predict the stage of Alzheimer's disease.<br>
**Stages:** Mild Demented, Moderate Demented, Non Demented, Very Mild Demented.

---

**Instructions:**
- Click 'Upload' or drag an MRI image.
- Click 'Submit' to see prediction.
- Try sample images below for quick demo.
"""

footer_md = """
---
Made by Nandhan | Powered by TensorFlow & Gradio
"""

iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=gr.Markdown(label="Prediction Result"),
    title="Alzheimer's Detection -- SWE1002 Project",
    description=description_md,
    examples=examples,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
    allow_flagging="never",
    article=footer_md
)

if __name__ == "__main__":
    iface.launch()