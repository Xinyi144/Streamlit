import os
import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from PIL import Image as PILImage

# Initialize OpenAI client (if needed)
openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])  # Replace with your actual key

# Initialize Google Generative AI client
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])  # Replace with your actual key

# Example of initializing a generative model with Google Generative AI
home_safety_model = genai.GenerativeModel(
    "gemini-1.5-flash",
    system_instruction="""
        You are a home safety assistant.
        User will upload an image of a room and you will analyze the image.
        Identify potential safety risks in the room based on the image.
        Provide general safety tips for baby-proofing the room.
        Output Contains:
        1. Description of potential risks found in the image.
        2. Safety tips based on the analysis.
        3. Suggestions for improving safety in the room.
    """
)

# Function for analyzing home safety
def home_safety_analysis(image):
    user_message = "Analyze the uploaded room image for safety risks."

    response = home_safety_model.generate_content(["Describe the risks in the image", image])

    return response.text.strip()

# Streamlit layout for uploading and displaying the image
st.title("Baby Home Safety Assistant")

# Upload an image
room_image = st.file_uploader("Upload a picture of a room", type=["jpg", "jpeg", "png"])

if room_image is not None:
    # Open and display the uploaded image
    img = PILImage.open(room_image)
    img.thumbnail((800, 800))  # Resize the image to max 800x800 pixels
    st.image(img, caption="Uploaded Room Image", use_column_width=True)

    # Analyze the image for safety risks
    if st.button("Analyze Room Safety"):
        try:
            result = home_safety_analysis(img)
            # Display the result from the home_safety_analysis function
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
else:
    st.write("Please upload a picture of a room to analyze safety risks.")