
import streamlit as st
from model import predict

st.set_page_config(page_title="Rumor Classifier", layout="centered")

st.title("🕵️ Rumor Detection App")

# Allow user to choose model folder and model type
model_dir = st.selectbox("Choose a model directory:", ["classification_models_text_comments", "classification_models_text_only"])
model_type = st.selectbox("Choose a model type:", ["BERT", "LSTM", "Transformer"])

st.write("Enter text to classify whether it is a **rumor** or **not**.")
user_input = st.text_area("Enter text here:", height=150)

if st.button("Predict"):
    if user_input.strip():
        prediction = predict(user_input, model_type=model_type, model_dir=model_dir)
        if prediction == "Rumor":
            st.success(f"🧠 Prediction using {model_type} from {model_dir}: **{prediction}**")
        else:
            st.error(f"🧠 Prediction using {model_type} from {model_dir}: **{prediction}**")
    else:
        st.warning("Please enter some text.")