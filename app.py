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

# ── UI ────────────────────────────────────────────────────
st.title("🔢 Handwritten Digit Recognizer")
st.markdown("**Draw a digit (0–9)** in the box below, then click **Predict**.")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("✏️ Draw Here")
    canvas = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas"
    )
    predict_btn = st.button("🔍 Predict", use_container_width=True)
    clear_note  = st.caption("Use the ↺ button on the canvas to clear and redraw.")

with col2:
    st.subheader("📊 Result")

    if predict_btn:
        if canvas.image_data is not None:
            # Convert canvas to 28x28 grayscale
            img = Image.fromarray(canvas.image_data.astype("uint8"))
            img = img.convert("L").resize((28, 28))
            img_array = np.array(img) / 255.0

            # Check if canvas is blank
            if img_array.max() < 0.1:
                st.warning("Canvas is empty. Please draw a digit first.")
            else:
                img_input = img_array.reshape(1, 28, 28, 1)
                prediction = model.predict(img_input, verbose=0)
                digit      = int(np.argmax(prediction))
                confidence = float(prediction[0][digit]) * 100

                st.metric(label="Predicted Digit", value=str(digit))
                st.metric(label="Confidence", value=f"{confidence:.2f}%")

                st.markdown("**Probability per digit:**")
                prob_dict = {str(i): float(prediction[0][i]) for i in range(10)}

                # Highlight bar for predicted digit
                for i in range(10):
                    prob  = float(prediction[0][i]) * 100
                    color = "#a29bfe" if i == digit else "#444"
                    bar   = int(prob * 1.5)
                    st.markdown(
                        f"""
                        <div style='display:flex; align-items:center; margin:2px 0'>
                            <span style='width:20px; font-weight:bold; color:{"#a29bfe" if i==digit else "white"}'>{i}</span>
                            <div style='height:16px; width:{bar}px; background:{color};
                                        border-radius:3px; margin:0 8px;'></div>
                            <span style='color:{"#a29bfe" if i==digit else "#aaa"};
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