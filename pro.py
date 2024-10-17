import os
import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from PIL import Image as PILImage
from streamlit_option_menu import option_menu

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])

# Initialize Gemini food_care_model
genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
food_care_model = genai.GenerativeModel("gemini-1.5-flash",
                                        system_instruction="""
                                You are a baby care assistant. 
                                User will upload an image of food or type in text format and select the age of the baby. 
                                You will receive either an image or a text description of food. 
                                Identify the food based on image or description, suitability of food (image or description) for a baby based on their age. 
                                You also should suggest the most suitable diets for the baby based on the age.
                                Output Contains:
                                1. Description Food(based on food image or description)
                                2. Food's Suitability level(scale from 1 to 3 based on baby's age, where 1 is most suitable and 3 is not suitable). Provide reasons in point form.
                                3. Suggestion Best Diets (suggested food description, quantity of suggested food, suitable timing for taking suggested food, and the expense of making suggested food based on age.
                                4. Alert (important things to be aware of.) If no, just ignored.
                                """)


# OpenAI food_care function
def food_care(prompt, age):
    system_prompt = """
    You are a baby care assistant. 
    User will upload an image of food or type in text format and select the age of the baby. 
    You will receive either an image or a text description of food. Identify the food based on image or description, suitability of food(image or description) for a baby based on their birth months. You also should suggest the most suitable diets for the baby based on the birth months.
    Output Contains:
    1. Description Food(based on food image or description)
    2. Food's Suitability level(state level based on birth's months). Provide reasons in point form.
    3. Suggestion Best Diets (suggested food description, quantity of suggested food, suitable timing for taking suggested food, and the expense of making suggested food based on birth's months.
    4. Alert (important things to be aware of.) If no, just ignored.
    """

    user_message = f"Food: {prompt}. Baby Age: {age} years/months."

    response = client.chat.completions.create(model='gpt-4o-mini',
                                              messages=[{
                                                  'role':
                                                  'system',
                                                  'content':
                                                  system_prompt
                                              }, {
                                                  'role': 'user',
                                                  'content': user_message
                                              }],
                                              temperature=1.0,
                                              max_tokens=200)
    return response.choices[0].message.content


# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the custom CSS
load_css("style.css")

# Sidebar navigation menu
with st.sidebar:
    selected = option_menu("Baby Care Assistant", [
        "Home", "Baby Food Care", "Home Safety", "Sleeping Monitor",
        "Voice Analysis", "Parenting Tips", "Calendar Reminder"
    ],
                           icons=[
                               "house", "check-circle", "shield-lock", "bed",
                               "microphone", "lightbulb", "calendar"
                           ],
                           default_index=0)
# Main content based on sidebar selection
if selected == "Home":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:180%;">Welcome to Baby Care Assistant</h1>',
        unsafe_allow_html=True)
    import time

    # Page title
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:180%;">Login Form</h1>',
        unsafe_allow_html=True)

    # Guardian details
    st.subheader('Guardian Information')
    guardian_name = st.text_input('Guardian Name')
    relationship = st.radio('Relationship with Baby', ['MOM', 'DAD', 'UNCLE', 'AUNTY', 'OTHERS'])
    email = st.text_input('Email Address')

    # Email validation
    if email and '@' not in email:
        st.error("Please enter a valid email address.")

    st.divider()

    # Baby details
    st.subheader('Baby Information')
    baby_name = st.text_input('Baby Name')
    gender = st.radio('Pick The Baby\'s Gender', ['Male', 'Female'])
    dob = st.date_input('Baby\'s Date of Birth')  # Date of Birth input
    age = st.slider('Baby Age (in Months)', 0, 48)

    # Religion selection
    religion = st.selectbox('Religion', ['Islam', 'Christianity', 'Hinduism', 'Buddhism', 'Others'])

    # Form submission
    if st.button('Submit'):
        # Simple validation
        if not guardian_name or not email or not baby_name:
            st.warning('Please fill in all required fields.')
        else:
            with st.spinner('Submitting...'):
                time.sleep(2)  # Simulate a delay for submission
            st.success(f'Form submitted successfully! \n Welcome {baby_name}!')
            st.balloons()

            # Optionally display submitted information
            st.write("### Submission Details")
            st.write(f"Guardian Name: {guardian_name}")
            st.write(f"Relationship with Baby: {relationship}")
            st.write(f"Email: {email}")
            st.write(f"Baby Name: {baby_name}")
            st.write(f"Gender: {gender}")
            st.write(f"Religion: {religion}")
            st.write(f"Date of Birth: {dob}")
            st.write(f"Age: {age} months")

elif selected == "Baby Food Care":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:180%;">Baby Food Care🍴</h1>',
        unsafe_allow_html=True)

    # Input type selection
    st.subheader("Select Input Method:")
    input_type = st.radio("Choose how you'd like to provide food information:",
                          ("Upload Image", "Type Text"))

    food_image = None
    food_description = None

    # Handle input based on user selection
    if input_type == "Upload Image":
        food_image = st.file_uploader("Upload a food image",
                                      type=["jpg", "jpeg", "png"])
        if food_image:
            img = PILImage.open(food_image)
            st.image(img, caption="Uploaded Food Image", use_column_width=True)
            imgDes = food_care_model.generate_content(["Describe image", img])
            food_description = imgDes.text
    elif input_type == "Type Text":
        food_description = st.text_input("Describe food:")

    # Input for baby's age
    st.subheader("Baby's Age")
    baby_age = st.number_input("Enter in months:",
                               min_value=0,
                               max_value=48,
                               step=1)

    # Submit button to run the function
    if st.button("Check Suitability"):
        if (food_image is not None or food_description) and baby_age > 0:
            try:
                result = food_care(food_description, baby_age)
                # Display the result from the food_care function
                st.success("Food Suitability Result:")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning(
                "Please provide an image/text description of food, then select baby's age!"
            )

elif selected == "Home Safety":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:180%;">Home Safety Tips</h1>',
        unsafe_allow_html=True)
    st.write(
        "Here you can find tips for ensuring a safe environment for your baby."
    )

elif selected == "Sleeping Monitor":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:180%;">Sleeping Monitor</h1>',
        unsafe_allow_html=True)
    st.write("Monitor your baby's sleep patterns and get insights.")

elif selected == "Voice Analysis":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:180%;">Voice Analysis</h1>',
        unsafe_allow_html=True)
    st.write("Analyze your baby's sounds and cries.")

elif selected == "Parenting Tips":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:180%;">Parenting Tips</h1>',
        unsafe_allow_html=True)
    st.write("Get useful tips and advice for parenting.")

elif selected == "Calendar Reminder":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:180%;">Calendar Reminder</h1>',
        unsafe_allow_html=True)
    st.write("Set reminders for important events related to your baby.")
