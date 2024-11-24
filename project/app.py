import streamlit as st
import pyttsx3
from PIL import Image
import pytesseract
from gtts import gTTS
from transformers import AutoProcessor, BlipForConditionalGeneration
import google.generativeai as genai
import os
import base64

# Set up Google Gemini API key (replace 'YOUR_API_KEY' with actual key)
GENAI_API_KEY = "AIzaSyD39FX94R8oA2veb0Cl03mdY6AYly-GEK4"
genai.configure(api_key=GENAI_API_KEY)

# Initialize BLIP model for image captioning
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Function to generate image caption using BLIP
def generate_image_caption(image):
    image = image.convert('RGB')
    inputs = processor(images=image, text="Describe the image", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Function to extract text from image using OCR (Tesseract)
def extract_text_from_image(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# Function to convert text to speech using gTTS
def text_to_speech(text):
    try:
        tts = gTTS(text, lang="en")
        audio_file = "output.mp3"
        tts.save(audio_file)
        return audio_file
    except Exception as e:
        return f"Error generating speech: {str(e)}"

# Streamlit configuration
st.set_page_config(page_title="Floral AI Assistant", page_icon="🌸", layout="wide")

# Custom CSS for floral decorations
st.markdown("""
    <style>
        body {
            background-color: #FFF4E6;
            font-family: 'Arial', sans-serif;
        }
        .main-title {
            color: #8B0000;
            font-family: 'Georgia', serif;
            text-align: center;
            font-size: 36px;
            margin-bottom: 10px;
        }
        .sub-title {
            color: #6B8E23;
            text-align: center;
            font-size: 24px;
            margin-bottom: 20px;
        }
        .flower-divider {
            text-align: center;
            margin: 20px 0;
        }
        .flower-divider img {
            width: 150px;
        }
        .content-box {
            border: 2px solid #8B0000;
            border-radius: 10px;
            background-color: #FFFAF0;
            padding: 20px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# App Title and Subtitle
st.markdown("<div class='main-title'>Floral AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Empowering users with AI-driven image descriptions and text-to-speech conversion.</div>", unsafe_allow_html=True)

# Decorative flower divider
st.markdown("<div class='flower-divider'><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Small_flower.svg/1024px-Small_flower.svg.png'></div>", unsafe_allow_html=True)

# Layout with two sections
col1, col2 = st.columns(2)

# Left Column: Image Description
with col1:
    st.markdown("<h3 style='text-align: center; color: #8B0000;'>🌸 Image Description</h3>", unsafe_allow_html=True)
    st.write("Upload an image, and we will generate a description of the scene, including actions, emotions, and visual elements.")
    
    uploaded_file = st.file_uploader("Upload an image for description...", type=["jpg", "jpeg", "png"], label_visibility="visible")

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Generating description..."):
            caption = generate_image_caption(image)
            st.subheader("Generated Caption:")
            st.write(caption)

        def generate_scene_description_with_gemini(caption):
            try:
                prompt = f"Generate an emotionally rich and action-based description of the following scene: {caption}"
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                ai_assistant = model.start_chat(history=[])
                response = ai_assistant.send_message(prompt)
                return response.text.strip() if response and response.text else "No description generated."
            except Exception as e:
                return f"Error generating description: {str(e)}"

        description = generate_scene_description_with_gemini(caption)
        if "Error" in description:
            st.error(description)
        else:
            st.subheader("Generated Description:")
            st.write(description)

# Right Column: OCR and Text-to-Speech
with col2:
    st.markdown("<h3 style='text-align: center; color: #8B0000;'>🌺 OCR and Text-to-Speech</h3>", unsafe_allow_html=True)
    st.write("Upload an image with text, and we will extract the text and convert it to speech.")

    ocr_uploaded_file = st.file_uploader("Upload an image with text...", type=["jpg", "jpeg", "png"], label_visibility="visible")

    if ocr_uploaded_file:
        image = Image.open(ocr_uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting text..."):
            text = extract_text_from_image(image)

        if text:
            st.subheader("Extracted Text:")
            st.write(text)

            with st.spinner("Converting text to speech..."):
                audio_file = text_to_speech(text)

            if os.path.exists(audio_file):
                st.subheader("Audio Playback:")
                audio = open(audio_file, "rb")
                st.audio(audio, format="audio/mp3")
                audio.close()
                os.remove(audio_file)
        else:
            st.warning("No text found in the image. Please try another image with visible text.")

# Footer
st.markdown("<div class='flower-divider'><img src='https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Small_flower.svg/1024px-Small_flower.svg.png'></div>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B8E23;'>🌸 Built with ❤ using Streamlit 🌸</p>", unsafe_allow_html=True)
