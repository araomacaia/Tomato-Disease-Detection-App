import os
import streamlit as st
from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Tomato Disease Detection Dashboard", layout="centered")

# -------------------------------------------------------
# Loading TFLite model
# -------------------------------------------------------
MODEL_PATH = os.path.join("tomato_disease_model", "tomato_model_flex.tflite")

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------------------------------
# Class Names
# -------------------------------------------------------
CLASS_NAMES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two_spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___healthy"
]

# -------------------------------------------------------
# Prediction Function
# -------------------------------------------------------
def predict_image(image_pil):
    img = image_pil.resize((224, 224))
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions) * 100)

    return predicted_class, confidence, dict(zip(CLASS_NAMES, predictions))

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.title("Tomato Disease Detection")
st.markdown("Upload a tomato leaf image to identify its health status or disease type.")

uploaded_file = st.file_uploader("Upload a tomato leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        predicted_class, confidence, _ = predict_image(image)

    st.success(f"*Prediction:* {predicted_class}")
    st.info(f"*Confidence:* {confidence:.2f}%")
# -------------------------------------------------------
# PAGE STYLE
# -------------------------------------------------------
page_style = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("{BACKGROUND_URL}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

.main-card {{
    background: rgba(255, 255, 255, 0.84);
    backdrop-filter: blur(6px);
    padding: 35px;
    border-radius: 18px;
    max-width: 900px;
    margin: 60px auto;
    box-shadow: 0 8px 25px rgba(0,0,0,0.25);
}}

.title {{
    text-align: center;
    font-size: 44px;
    color: #b71c1c;
    font-family: 'Trebuchet MS', sans-serif;
    font-weight: 800;
    text-shadow: 2px 2px 8px rgba(255,255,255,0.9);
    margin-bottom: 10px;
}}

.subtitle {{
    text-align: center;
    font-size: 20px;
    font-weight: 700;
    color: #111;
    margin-bottom: 30px;
    text-shadow: 1px 1px 4px rgba(255,255,255,0.6);
}}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='title'>Tomato Disease Detection & Classification Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect tomato leaf diseases and visualize AI confidence metrics using your trained model</div>", unsafe_allow_html=True)

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------
st.markdown("<h5 style='font-weight:700; font-size:19px; text-align:center; color:#0f172a;'>Upload a tomato leaf image</h5>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf Image", width=400)
    st.success("✅ Image uploaded successfully!")

    # Running prediction
    label, confidence, probs = predict_image(img)

    # Displaying result container
    st.markdown(f"""
    <div style="
        background-color: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        border-radius: 18px;
        padding: 20px 30px;
        margin: 20px auto;
        width: 90%;
        max-width: 900px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    ">
        <div style="font-size: 22px; color: #b71c1c; font-weight: bold;">
            🩺 Disease Detected: <b>{label}</b>
        </div>
        <div style="font-size: 20px; color: #1e293b; font-weight: bold;">
            Confidence: <b>{confidence:.2f}%</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability chart with red highlight for detected class
    df = pd.DataFrame({
        "Class": list(probs.keys()),
        "Confidence": list(probs.values())
    }).sort_values("Confidence", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(df["Class"], df["Confidence"], height=0.5)

    # Highlight detected disease in red
    for bar, cls in zip(bars, df["Class"]):
        if cls == label:
            bar.set_color("red")
        else:
            bar.set_color("gray")

    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Confidence per Class")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    if st.button("🔁 Clear and Upload Another Image"):
        st.experimental_rerun()
else:
    st.info("Please upload a tomato leaf image to begin the analysis.")

# -------------------------------------------------------
# FOOTER
# -------------------------------------------------------
footer_html = f"""
<div style="
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: rgba(255,255,255,0.95);
    padding: 10px 20px;
    border-radius: 40px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
">
    <img src="{logo_url}" alt="Logo" style="
        width: 50px;
        height: 50px;
        border-radius: 50%;
        object-fit: cover;
        box-shadow: 0 0 6px rgba(0,0,0,0.3);
    ">
    <small style="font-size: 14px; color: #222;">
        <b>Developed by Arao Zau Macaia</b> | Elevate Labs AI & ML Intern | NIT Durgapur
    </small>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)