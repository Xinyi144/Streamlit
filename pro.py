import os
import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from PIL import Image as PILImage
from streamlit_option_menu import option_menu
import time
import pandas as pd
from datetime import datetime, timedelta

# Define CSS to change the background color
background_color_css = """
<style>
/* Adjust the main content area */
.stApp {
    background: linear-gradient(135deg, #1D3557,#22577A);
}
</style>
"""
st.markdown(background_color_css, unsafe_allow_html=True)

# Define CSS for dialogue style
dialogue_css = """
<style>
.dialogue {
    font-family: 'Times New Roman', serif;
    font-size: 280%;
    background-color: #231942;
    padding: 20px;
    border-radius: 15px;
    border: 5px solid black;
    position: relative;
    display: inline-block;
    color: white;
    margin: 20px;
}

.dialogue:before {
    content: "";
    position: absolute;
    bottom: 0;
    left: 20px;
    width: 0;
    height: 0;
    border: 20px solid transparent;
    border-top-color: #231942;
    border-bottom: 0;
    border-left: 0;
    margin-left: -10px;
    margin-bottom: -20px;
}
</style>
"""

#client = OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
#genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])

# Function to load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Load the custom CSS
load_css("style.css")

# Sidebar navigation menu
with st.sidebar:
    selected = option_menu("Baby Care Assistant", [
        "Sign Up", 
        "Login", 
        "Baby Food Care", 
        "Home Safety", 
        "Voice Analysis", 
        "Parenting Tips", 
        "Calendar Reminder",
        "Sleeping Monitor",
    ],
    icons=["person-add", "person", "check-circle", "shield-lock", "mic", "lightbulb", "calendar","moon",],  # Updated icons
    default_index=0)


# SIGNUP PAGE--------------------------------------------------
if selected == "Sign Up":
    st.markdown(dialogue_css, unsafe_allow_html=True)
    st.markdown('<div class="dialogue">Welcome to Baby Care Assistant</div>', unsafe_allow_html=True)
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:250%; color:#CAF0F8; text-decoration:underline;">SIGN UP</h1>',
        unsafe_allow_html=True
    )

    # Initialize session state for storing user data
    if 'user_data' not in st.session_state:
        st.session_state.user_data={}

    def calculate_age_in_months(dob):
        """Calculate age in months based on the date of birth."""
        today = datetime.today()
        age_in_months = (today.year - dob.year) * 12 + today.month - dob.month
        return age_in_months
    
    def sign_up():
        st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Mummy Information:</p>',unsafe_allow_html=True)
        guardian_name = st.text_input("Guardian Name")
        relationship = st.selectbox("Relationship with Baby", ["Mom", "Dad", "Uncle", "Aunty", "Others"])
        email = st.text_input("Email Address")

        # Baby Information
        st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Baby Information:</p>',unsafe_allow_html=True)
        baby_name = st.text_input("Baby Name")
        baby_gender = st.selectbox("Baby Gender", ["Male", "Female", "Other"])
        date_of_birth = st.date_input("Date of Birth")

        if st.button("Sign Up"):
            # Check if email already exists
            if email in st.session_state.user_data:
                st.error("Email already registered. Please log in.")
            else:
                # Calculate age in months
                age_in_months = calculate_age_in_months(date_of_birth)

                # Store user data in the session state dictionary
                st.session_state.user_data[email] = {
                    'guardian_name': guardian_name,
                    'relationship': relationship,
                    'baby_name': baby_name,
                    'baby_gender': baby_gender,
                    'date_of_birth': date_of_birth,
                    'month': date_of_birth.strftime("%B"),
                    'age_in_months': age_in_months  # Store calculated age in months
                }
                st.success("Sign Up Successful! You can now log in.")
                st.balloons()  # Add balloon effect

    st.write(sign_up())

# LOGIN PAGE------------------------------------------------------
elif selected=="Login":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:250%; color:#CAF0F8; text-decoration:underline;">LOGIN</h1>',
        unsafe_allow_html=True
    )

    # Initialize session state for storing user data
    if 'user_data' not in st.session_state:
        st.session_state.user_data={}
    
    def login():
        email = st.text_input("Email Address")

        if st.button("Login"):
            if email in st.session_state.user_data:
                user_info = st.session_state.user_data[email]
                st.success(f"Login Successful! Welcome back, {user_info['guardian_name']}!")
                st.write(f"Baby Name: {user_info['baby_name']}")
                st.write(f"Baby Gender: {user_info['baby_gender']}")
                st.write(f"Date of Birth: {user_info['date_of_birth']}")
                st.write(f"Month: {user_info['month']}")
                st.write(f"Age: {user_info['age_in_months']} months")  # Display age in months
                st.balloons()  # Add balloon effect
            else:
                st.error("Email not found. Please sign up first.")

    st.write(login())


# BABY FOOD CARE PAGE-----------------------------------------
elif selected == "Baby Food Care":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:250%; color:#CAF0F8; text-decoration:underline;">BABY FOOD CARE</h1>',
        unsafe_allow_html=True
    )


    # Initialize Gemini food_care_model
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

        user_message = f"Food: {prompt}. Baby Age: {age} months."

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
                                                  max_tokens=300)
        return response.choices[0].message.content

    # Input type selection
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Select Input Method:</p>',unsafe_allow_html=True)
    input_type = st.radio("Choose how you'd like to provide food information:",
                          ("Upload Image", "Type Text"))

    food_image = None
    food_description = None
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
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Baby\'s Age:</p>',unsafe_allow_html=True)
    baby_age = st.number_input("Enter in months:",
                               min_value=0,
                               max_value=48,
                               step=1)

    if st.button("Check Suitability"):
        if (food_image is not None or food_description) and baby_age > 0:
            try:
                result = food_care(food_description, baby_age)
                st.success("Food Suitability Result:")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning(
                "Please provide an image/text description of food, then select baby's age!"
            )

# HOME SAFETY PAGE---------------------------------------
elif selected == "Home Safety":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:250%; color:#CAF0F8; text-decoration:underline;">BABY HOME SAFETY </h1>',
        unsafe_allow_html=True
    )

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

    def home_safety_analysis(image):
        user_message = "Analyze the uploaded room image for safety risks."

        response = home_safety_model.generate_content(["Describe the risks in the image", image])

        return response.text.strip()
        
    room_image = st.file_uploader("Upload a picture of a room", type=["jpg", "jpeg", "png"])

    if room_image is not None:
        img = PILImage.open(room_image)
        img.thumbnail((800, 800))  # Resize the image to max 800x800 pixels
        st.image(img, caption="Uploaded Room Image", use_column_width=True)

        if st.button("Analyze Room Safety"):
            try:
                result = home_safety_analysis(img)
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.write("Please upload a picture of a room to analyze safety risks.")

# SLEEPING MONITOR PAGE-----------------------------------------
elif selected == "Sleeping Monitor":
    st.markdown('<h1 style="font-family:\'Times New Roman\'; font-size:250%; color:#CAF0F8; text-decoration:underline;">SLEEPING MONITOR</h1>', unsafe_allow_html=True)

    baby_sleeping_model = genai.GenerativeModel("gemini-1.5-flash",
                                  system_instruction="""
                                    You are a baby care assistant. 
                                    You need to analyze the baby's sleeping pattern based on the input information (sleeping position of the baby, age, and sleeping time).
                                    Assess whether this sleeping pattern is good for the baby or not. If it is bad, list the disadvantages and suggest solutions to improve the sleeping pattern. 
                                    Based on the sleeping time provided, assess whether it is appropriate for the baby's age, and if not, suggest the best sleeping time.
                                  """
                                 )
    
    # OpenAI baby sleeping function
    def baby_sleeping(age, sleeping_time, prompt):
        system_prompt = """
        You are a baby care assistant. 
        You need to analyze the baby's sleeping pattern based on the input information (sleeping position of the baby, age, and sleeping time).
        Assess whether this sleeping pattern is good for the baby or not. If it is bad, list the disadvantages and suggest solutions to improve the sleeping pattern. 
        Based on the sleeping time provided, assess whether it is appropriate for the baby's age, and if not, suggest the best sleeping time.
        """
    
        user_message = f"Baby Age: {age} years/months. Sleeping Time: {sleeping_time}. Sleeping Pattern: {prompt}. "
    
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                  {'role': 'system', 'content': system_prompt},
                  {'role': 'user', 'content': user_message}
            ],
            temperature=1.0,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    # Baby Sleeping Analyst Title
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Baby Sleeping Analyst:</p>',unsafe_allow_html=True)
    
    # Handle input based on user selection
    uploaded_image = st.file_uploader("Choose a sleeping position image", type=["jpg", "jpeg", "png"])
    
    # Display the uploaded image
    if uploaded_image:
        img = PILImage.open(uploaded_image)
        img.thumbnail((800, 800))  # Resize image for display
        st.image(img, caption="Uploaded Baby Sleeping Position", use_column_width=True)
    
        # Generate description from image (make sure this part is compatible with the Gemini API)
        img_description = baby_sleeping_model.generate_content(["Describe the image", img])
        baby_pattern_description = img_description.text  # This should be the text description of the sleeping pattern
    
    # Input for baby's age
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Baby\'s Age:</p>',unsafe_allow_html=True)
    baby_age = st.number_input("in months:", min_value=0, max_value=48, step=1)
    
    # Input for sleeping time
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Sleeping Time:</p>',unsafe_allow_html=True)
    baby_sleeping_time = st.number_input("in hours:", min_value=0, max_value=24, step=1)
    
    # Submit button to run the function
    if st.button("Check Sleep"):
        if (uploaded_image is not None) and (baby_age > 0):
            try:
                # Use the generated description if an image is uploaded
                result = baby_sleeping(baby_age, baby_sleeping_time, baby_pattern_description)
    
                # Display the result from the baby sleeping function
                st.write(result)
            except Exception as e:
                st.write("An error occurred:", str(e))
        else:
            st.write("Please provide an image of the baby and their age.")

# VOICE ANALYSIS PAGE-----------------------------------------------
elif selected == "Voice Analysis":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:250%; color:#CAF0F8; text-decoration:underline;">VOICE ANALYSIS</h1>',
        unsafe_allow_html=True
    )

# PARENTING TIPS PAGE------------------------------------------------
elif selected == "Parenting Tips":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:250%; color:#CAF0F8; text-decoration:underline;">PARENTING TIPS</h1>',
        unsafe_allow_html=True
    )

    parenting_tips_model = genai.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction="""
            You are a parenting assistant. 
            User will upload an image of baby preparations, and you will analyze the image. 
            Based on the image, provide parenting tips relevant to the items in the image.
            Output Contains:
            1. Recognize the items in the image (e.g., crib, baby clothes, stroller).
            2. Provide relevant parenting tips and advice based on the identified items.
            3. Suggest additional items or preparations if something is missing.
        """
    )

    # Function for analyzing baby preparations
    def analyze_baby_preparation(image):
        user_message = "Analyze the uploaded image of baby preparations."
        response = parenting_tips_model.generate_content([user_message, image])
        return response.text.strip()

    # Streamlit layout for uploading and displaying the image
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:180%;font-weight:bold;">Parenting Assistant:</p>',unsafe_allow_html=True)

    # Dictionary to hold all recommendations and user inputs
    user_data = {}

    # Ask for the number of months pregnant
    months_pregnant = st.number_input("How many months are you pregnant?", min_value=0, max_value=9, step=1)

    # Recommendations based on the number of months pregnant
    recommendations = {
        1: ["Prenatal vitamins", "First prenatal visit", "Pregnancy journal"],
        2: ["Choose a healthcare provider", "Discuss birth plan", "Baby names"],
        3: ["Maternity clothes", "Childbirth classes", "Prepare home for baby"],
        4: ["Crib or bassinet", "Baby registry", "Baby clothes"],
        5: ["Feeding supplies (bottles, bibs)", "Baby shower planning", "Research pediatricians"],
        6: ["Stroller", "Car seat", "Diapers and wipes"],
        7: ["Pack hospital bag", "Finish baby registry", "Baby-proofing supplies"],
        8: ["Install car seat", "Arrange help after birth", "Tour birthing facility"],
        9: ["Prepare for labor", "Check essentials", "Enjoy final days"]
    }

    # Initialize user_data for the selected month
    if months_pregnant not in user_data:
        user_data[months_pregnant] = {
            "recommendations": [],
            "user_input": ""
        }

    # Generate the recommendations based on the pregnancy month
    if months_pregnant > 0:
        st.header("Recommendations Based on Your Pregnancy Month")
        st.write("Here are some things you should consider buying:")
        for item in recommendations[months_pregnant]:
            st.write(f"- {item}")
            user_data[months_pregnant]["recommendations"].append(item)

    # Upload an image of baby preparations
    uploaded_image = st.file_uploader("Upload a picture of your baby preparations", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Open and display the uploaded image
        img = PILImage.open(uploaded_image)
        img.thumbnail((800, 800))  # Resize the image to max 800x800 pixels
        st.image(img, caption="Uploaded Baby Preparation Image", use_column_width=True)

        # Analyze the image for parenting tips
        if st.button("Analyze Baby Preparation"):
            try:
                result = analyze_baby_preparation(img)
                st.write(result)  # Display the result from the parenting tips analysis function
                user_data[months_pregnant]["user_input"] = result  # Store the analysis result
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.write("Please upload a picture of your baby preparations.")

    # Divider for exercise recommendations
    st.divider()
    # Ask if the user needs exercise recommendations
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:180%;font-weight:bold;">Do you need any exercise recommendations for yourself as a mom?</p>',unsafe_allow_html=True)
    exercise_needed = st.radio("", ("Yes", "No"))  # No label for the radio buttons

    # Provide exercise recommendations based on pregnancy month if they say yes
    if exercise_needed == "Yes":
        st.markdown('<p style="font-family:\'Times New Roman\'; font-size:150%;">Exercise Recommendations: </p>', unsafe_allow_html=True)

        exercises = {
            1: ["Walking", "Gentle stretching", "Pelvic tilts"],
            2: ["Walking", "Prenatal yoga", "Swimming"],
            3: ["Walking", "Low-impact aerobics", "Bodyweight exercises"],
            4: ["Walking", "Pregnancy yoga", "Kegel exercises"],
            5: ["Walking", "Pilates for pregnancy", "Light resistance training"],
            6: ["Walking", "Water aerobics", "Stretching"],
            7: ["Walking", "Prenatal dance", "Gentle strength training"],
            8: ["Walking", "Breathing exercises", "Pelvic floor exercises"],
            9: ["Gentle walking", "Relaxation exercises", "Preparation for labor exercises"]
        }

        if months_pregnant in exercises:
            st.header("Here are some exercises you can do this month:")
            for exercise in exercises[months_pregnant]:
                st.write(f"- {exercise}")

    # Input text or image section
    st.divider()
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:180%;font-weight:bold;">Check list that already bought for Baby:</p>',unsafe_allow_html=True)
    st.write("Please let us know if you have prepared everything adequately!")

    # Text input for user feedback
    user_input = st.text_area("What else do you think should be prepared?", height=150)

    # Save user input
    if user_input:
        user_data[months_pregnant]["user_input"] += f"\n{user_input}"  # Append user input for the month

    # Display all recommendations and user input for the selected month
    if user_data[months_pregnant]["recommendations"]:
        st.header("Your Prepared Items and Suggestions:")
        st.write("### Recommendations:")
        for item in user_data[months_pregnant]["recommendations"]:
            st.write(f"- {item}")

        st.write("### Your Prepared Item(s):")
        st.write(user_data[months_pregnant]["user_input"])
        
# CALENDER REMINDER-----------------------------------------
elif selected == "Calendar Reminder":
    st.markdown(
        '<h1 style="font-family:\'Times New Roman\'; font-size:250%; color:#CAF0F8; text-decoration:underline;">FEEDING & VACCINATION TRACKER</h1>', unsafe_allow_html=True)
    st.markdown("Set reminders for important events related to your baby.")

    # Create an empty DataFrame to store feeding and vaccination records
    if 'records' not in st.session_state:
        st.session_state.records = pd.DataFrame(columns=["Type", "Date", "Next Time"])

    # Initialize session state for time inputs
    if 'feeding_time' not in st.session_state:
        st.session_state.feeding_time = datetime.now().time()
    if 'vaccination_time' not in st.session_state:
        st.session_state.vaccination_time = datetime.now().time()

    # Function to calculate the next feeding time
    def calculate_next_feeding(current_time, interval_hours):
        return current_time + timedelta(hours=interval_hours)

    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Feeding Schedule:</p>',unsafe_allow_html=True)
    feeding_time_input = st.time_input("Enter feeding time (HH:MM)", value=st.session_state.feeding_time)
    feeding_interval = st.number_input("Enter feeding interval (in hours)", min_value=1, max_value=12, step=1)

    if st.button("Add Feeding"):
        current_time = datetime.combine(datetime.today(), feeding_time_input)
        next_time = calculate_next_feeding(current_time, feeding_interval)
        new_record = pd.DataFrame({"Type": ["Feeding"], "Date": [current_time], "Next Time": [next_time]})
        st.session_state.records = pd.concat([st.session_state.records, new_record], ignore_index=True)
        # Update the feeding time in session state
        st.session_state.feeding_time = feeding_time_input

        # Show the next feeding time to the user
        st.success(f"The next feeding is scheduled at: {next_time.strftime('%H:%M')}")

    st.divider()
    # Input section for vaccination details
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Vaccination Schedule:</p>',unsafe_allow_html=True)
    vaccination_date_input = st.date_input("Enter vaccination date", value=datetime.today())
    vaccination_time_input = st.time_input("Enter vaccination time (HH:MM)", value=st.session_state.vaccination_time)

    if st.button("Add Vaccination"):
        vaccination_time = datetime.combine(vaccination_date_input, vaccination_time_input)
        new_record = pd.DataFrame({"Type": ["Vaccination"], "Date": [vaccination_time], "Next Time": ["N/A"]})
        st.session_state.records = pd.concat([st.session_state.records, new_record], ignore_index=True)
        # Update the vaccination time in session state
        st.session_state.vaccination_time = vaccination_time_input

    st.divider()
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Records:</p>',unsafe_allow_html=True)
    st.write(st.session_state.records)

    # Provide reminders
    if st.session_state.records.shape[0] > 0:
        st.markdown('<p style="font-family:\'Times New Roman\'; font-size:180%;font-weight:bold;">Reminders</p>',unsafe_allow_html=True)
        now = datetime.now()
        for index, row in st.session_state.records.iterrows():
            if row["Type"] == "Feeding":
                next_feed_time = row["Next Time"]
                st.write(f"Next feeding is scheduled at: {next_feed_time.strftime('%Y-%m-%d %H:%M')}")
                # Check if the feeding time is now or in the next few minutes
                if now >= next_feed_time - timedelta(minutes=5) and now < next_feed_time:
                    st.warning(f"Reminder: It's almost time to feed! Next feeding is at: {next_feed_time.strftime('%H:%M')}.")
            elif row["Type"] == "Vaccination":
                st.write(f"Vaccination scheduled at: {row['Date'].strftime('%Y-%m-%d %H:%M')}")
                # Check if the vaccination time is now or in the next few minutes
                if now >= row["Date"] - timedelta(minutes=5) and now < row["Date"]:
                    st.warning(f"Reminder: It's almost time for vaccination at: {row['Date'].strftime('%H:%M')}.")

    st.divider()
    # Option to download the records as a CSV file
    st.markdown('<p style="color:#00B4D8; font-family:\'Times New Roman\'; font-size:200%;font-weight:bold;">Click to Download Records:</p>',unsafe_allow_html=True)
    csv = st.session_state.records.to_csv(index=False).encode('utf-8')
    st.download_button("Download Records as CSV", csv, "feeding_vaccination_records.csv", "text/csv", key='download-csv')


