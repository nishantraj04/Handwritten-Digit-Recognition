import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.h5")

model = load_model()

# ── Preprocessing ─────────────────────────────────────────
def preprocess(image_data):
    """
    Mimics MNIST preprocessing:
    1. Convert to grayscale
    2. Find bounding box of drawn digit
    3. Crop, pad, center, and resize to 28x28
    """
    img = Image.fromarray(image_data.astype("uint8")).convert("L")
    img_np = np.array(img)

    # Check if canvas is blank
    if img_np.max() < 10:
        return None

    # Find bounding box of non-black pixels
    rows = np.any(img_np > 10, axis=1)
    cols = np.any(img_np > 10, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop to bounding box
    cropped = img_np[rmin:rmax + 1, cmin:cmax + 1]

    # Add 40% padding around digit (like MNIST centering)
    h, w = cropped.shape
    pad = int(max(h, w) * 0.4)
    padded = np.pad(cropped, pad, mode='constant', constant_values=0)

    # Make square
    h2, w2 = padded.shape
    size = max(h2, w2)
    square = np.zeros((size, size), dtype=np.uint8)
    y_off = (size - h2) // 2
    x_off = (size - w2) // 2
    square[y_off:y_off + h2, x_off:x_off + w2] = padded

    # Resize to 28x28 (MNIST size)
    final = Image.fromarray(square).resize((28, 28), Image.LANCZOS)
    return np.array(final) / 255.0

# ── UI ────────────────────────────────────────────────────
st.title("🔢 Handwritten Digit Recognizer")
st.markdown("**Draw a digit (0–9)** in the box below, then click **Predict**.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✏️ Draw Here")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=22,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    predict_btn = st.button("🔍 Predict", use_container_width=True)
    st.caption("Tips: Draw large, centered, and thick. Use ↺ to clear.")

with col2:
    st.subheader("📊 Result")

    if predict_btn:
        if canvas.image_data is not None:
            processed = preprocess(canvas.image_data)

            if processed is None:
                st.warning("Canvas is empty. Please draw a digit first.")
            else:
                img_input  = processed.reshape(1, 28, 28, 1)
                prediction = model.predict(img_input, verbose=0)
                digit      = int(np.argmax(prediction))
                confidence = float(prediction[0][digit]) * 100

                st.metric(label="Predicted Digit", value=str(digit))
                st.metric(label="Confidence", value=f"{confidence:.2f}%")

                st.markdown("**Probability per digit:**")
                for i in range(10):
                    prob  = float(prediction[0][i]) * 100
                    color = "#a29bfe" if i == digit else "#444"
                    bar   = int(prob * 1.5)
                    st.markdown(
                        f"""
                        <div style='display:flex; align-items:center; margin:2px 0'>
                            <span style='width:20px; font-weight:bold;
                                color:{"#a29bfe" if i == digit else "white"}'>{i}</span>
                            <div style='height:16px; width:{bar}px; background:{color};
                                border-radius:3px; margin:0 8px;'></div>
                            <span style='color:{"#a29bfe" if i == digit else "#aaa"};
                                font-size:13px'>{prob:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.warning("Please draw a digit first.")
    else:
        st.info("Draw a digit on the left and click **Predict**.")

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:13px'>"
    "CNN trained on 60,000 MNIST samples · 99.01% test accuracy · "
    "Built by <a href='https://github.com/nishantraj04' style='color:#a29bfe'>nishantraj04</a>"
    "</p>",
    unsafe_allow_html=True
)