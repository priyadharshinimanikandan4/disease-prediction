import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import streamlit as st
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


import streamlit as st
import hashlib
import sqlite3
import os
#  Ensure session state variables are initialized
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""  # Initialize with an empty string

# Database setup and functions
conn = sqlite3.connect("users.db")
c = conn.cursor()

# Create the prediction history table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    disease_type TEXT,
    prediction_result TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)''')
conn.commit()

# Function to fetch prediction history
def get_prediction_history(username):
    c.execute("SELECT * FROM prediction_history WHERE username=?", (username,))
    return c.fetchall()

# Function to save prediction history
def save_prediction_history(username, disease_type, prediction_result):
    c.execute("INSERT INTO prediction_history (username, disease_type, prediction_result) VALUES (?, ?, ?)",
              (username, disease_type, prediction_result))
    conn.commit()
# Create a database for storing users

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to check user credentials
def check_login(username, password):
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    if result:
        return result[0] == hash_password(password)
    return False

# Function to add a new user
def add_user(username, password):
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
    conn.commit()

# Ensure session state exists

# Logout function
def logout():
    st.session_state["authenticated"] = False
    st.experimental_rerun()

# 🎨 Apply persistent styles (prevents style reset on rerun)
def apply_custom_styles():
    st.markdown(
        """
        <style>
        /* Target the main container */
        .stApp {
            background-color: #e6f7ff;  /* Light blue background */
        }

        /* Ensure other elements are visible */
        .stMarkdown, .stTextInput, .stNumberInput, .stButton, .stImage {
            color: #333333;  /* Dark text for readability */
        }

        /* Sidebar styles */
        .stSidebar {
            background-color: #2c3e50 !important;  /* Dark sidebar */
            color: white !important;
        }

        /* Button styles */
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 10px;
            padding: 10px;
            font-weight: bold;
        }

        /* Input field styles */
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            border-radius: 5px;
            padding: 8px;
            border: 1px solid #4CAF50 !important;
        }

        /* Image styles */
        .stImage>img {
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )
apply_custom_styles()
logo = Image.open('download (3).jpg')  # Replace 'logo.png' with the path to your logo image

# Display the logo at the top of the page
st.image(logo, width=200)  # Adjust the width as needed

# Your existing code continues here...


# 🚀 If user is not authenticated, show Login & Signup
if not st.session_state["authenticated"]:
    st.title("🔒 Welcome to HealthGuard Pro")
    
    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Signup"])

    # 🔹 Login Form
    with tab1:
        st.subheader("Login to Your Account")
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.session_state["authenticated"] = True
                st.success(f"✅ Welcome, {username}!")
                st.experimental_rerun()  # Refresh the page to hide login
            else:
                st.error("🚫 Incorrect Username or Password")

    # 🔹 Signup Form
    with tab2:
        st.subheader("Create a New Account")
        new_username = st.text_input("👤 Choose a Username")
        new_password = st.text_input("🔑 Choose a Password", type="password")
        confirm_password = st.text_input("🔄 Confirm Password", type="password")

        if st.button("Signup"):
            if new_password == confirm_password:
                try:
                    add_user(new_username, new_password)
                    st.success("✅ Account created successfully! Please log in.")
                except sqlite3.IntegrityError:
                    st.error("⚠️ Username already exists. Try another one.")
            else:
                st.error("🚫 Passwords do not match!")

    st.stop()  #  Prevents disease prediction UI from showing when not logged in

# 🎉 User is logged in, show Disease Prediction System
st.sidebar.title(f"👋 Welcome, {st.session_state['username']}!")
st.sidebar.button("🔴 Logout", on_click=logout)

# loading the models
diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
parkinson_model = joblib.load("models/parkinsons_model.sav")
# Load the lung cancer prediction model
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')

# Load the pre-trained model
breast_cancer_model = joblib.load('models/breast_cancer.sav')

# Load the pre-trained model
chronic_disease_model = joblib.load('models/chronic_model.sav')

# Load the hepatitis prediction model
hepatitis_model = joblib.load('models/hepititisc_model.sav')


liver_model = joblib.load('models/liver_model.sav')# Load the lung cancer prediction model
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')

# Session state for user authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = None  #
# 🚀 If user is not authenticated, show Login & Signup
if not st.session_state["authenticated"]:
    st.title("🔒 Welcome to Health Prediction System")

    tab1, tab2 = st.tabs(["🔑 Login", "🆕 Signup"])

    # 🔹 Login Form
    with tab1:
        st.subheader("Login to Your Account")
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.session_state["authenticated"] = True
                st.session_state["username"] = username  # Store username in session state
                st.success(f"✅ Welcome, {username}!")
                st.experimental_rerun()  # Refresh the page to hide login
            else:
                st.error("🚫 Incorrect Username or Password")

    # 🔹 Signup Form
    with tab2:
        st.subheader("Create a New Account")
        new_username = st.text_input("👤 Choose a Username")
        new_password = st.text_input("🔑 Choose a Password", type="password")
        confirm_password = st.text_input("🔄 Confirm Password", type="password")

        if st.button("Signup"):
            if new_password == confirm_password:
                try:
                    add_user(new_username, new_password)
                    st.success("✅ Account created successfully! Please log in.")
                except sqlite3.IntegrityError:
                    st.error("⚠️ Username already exists. Try another one.")
            else:
                st.error("🚫 Passwords do not match!")

    st.stop()  # ❌ Prevents disease prediction UI from showing when not logged in



# sidebar
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction', [
        'Disease Prediction',
        'Diabetes Prediction',
        'Heart Disease Prediction',
        'Parkinson Prediction',
        'Liver Prediction',
        'Hepatitis Prediction',
        'Lung Cancer Prediction',
        'Chronic Kidney Prediction',
        'Breast Cancer Prediction',
        'Prediction History',
        'AI Chatbot',
        'Book Consultation',
        'Health Tips',
        'Health Trends',
        'Feedback',
        'Achievements',
    ],
    icons=['activity', 'heart', 'person', 'person', 'person', 'person', 'bar-chart-fill', 'chat', 'calendar', 'book', 'graph-up', 'megaphone', 'trophy'],
    default_index=0)
        




# multiple disease prediction
if selected == 'Disease Prediction': 
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    # Title
    st.write('# Disease Prediction using Machine Learning')
    image = Image.open('download (8).jpg')
    st.image(image, caption='Disease prediction')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'): 
        # Run the model with the python script
        
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')


        tab1, tab2= st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')





# Diabetes prediction page
if selected == 'Diabetes Prediction':
    st.title("Diabetes Disease Prediction")
    image = Image.open('images.jpg')
    st.image(image, caption='diabetes disease prediction')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Number of Pregnencies")
    with col2:
        Glucose = st.number_input("Glucose level")
    with col3:
        BloodPressure = st.number_input("Blood pressure value")
    with col1:
        SkinThickness = st.number_input("Skin thickness value")
    with col2:
        Insulin = st.number_input("Insulin value")
    with col3:
        BMI = st.number_input("BMI value")
    with col1:
        DiabetesPedigreeFunction = st.number_input("Diabetes pedigree function value")
    with col2:
        Age = st.number_input("AGE")

    # Code for prediction
    diabetes_dig = ''

    # Button
    if st.button("Diabetes test result"):
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        )

        # Determine the result
        if diabetes_prediction[0] == 1:
            result = "Diabetic"
            diabetes_dig = "We are really sorry to say but it seems like you are Diabetic."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            result = "Not Diabetic"
            diabetes_dig = "Congratulations, you are not diabetic."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        # Save prediction history
        save_prediction_history(st.session_state["username"], "Diabetes", result)

        # Display result
        st.success(f"{name}, {diabetes_dig}")


# Heart prediction page

if selected == 'Heart Disease Prediction':
    st.title("Heart Disease Prediction")
    image = Image.open('download (4).jpg')
    st.image(image, caption='heart failure')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")
    with col2:
        sex = st.selectbox("Gender", ["Male", "Female"])
        sex = 1 if sex == "Male" else 0
    with col3:
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        cp = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp)
    with col1:
        trestbps = st.number_input("Resting Blood Pressure")
    with col2:
        chol = st.number_input("Serum Cholesterol")
    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        fbs = 1 if fbs == "Yes" else 0
    with col1:
        restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        restecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(restecg)
    with col2:
        thalach = st.number_input("Max Heart Rate Achieved")
    with col3:
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
        exang = 1 if exang == "Yes" else 0
    with col1:
        oldpeak = st.number_input("ST Depression Induced by Exercise")
    with col2:
        slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        slope = ["Upsloping", "Flat", "Downsloping"].index(slope)
    with col3:
        ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy")
    with col1:
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
        thal = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

    # Code for prediction
    heart_dig = ''

    # Button
    if st.button("Heart test result"):
        heart_prediction = heart_model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        )

        # Determine the result
        if heart_prediction[0] == 1:
            result = "Heart Disease"
            heart_dig = "We are really sorry to say but it seems like you have Heart Disease."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            result = "No Heart Disease"
            heart_dig = "Congratulations, you don't have Heart Disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        # Save prediction history
        save_prediction_history(st.session_state["username"], "Heart Disease", result)

        # Display result
        st.success(f"{name}, {heart_dig}")






if selected == 'Parkinson Prediction':
    st.title("Parkinson's Disease Prediction")
    image = Image.open('download (5).jpg')
    st.image(image, caption='Parkinson\'s disease')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        MDVP_Fo = st.number_input("MDVP:Fo(Hz)")
    with col2:
        MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)")
    with col3:
        MDVP_Flo = st.number_input("MDVP:Flo(Hz)")
    with col1:
        MDVP_Jitter = st.number_input("MDVP:Jitter(%)")
    with col2:
        MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)")
    with col3:
        MDVP_RAP = st.number_input("MDVP:RAP")
    with col1:
        MDVP_PPQ = st.number_input("MDVP:PPQ")
    with col2:
        Jitter_DDP = st.number_input("Jitter:DDP")
    with col3:
        MDVP_Shimmer = st.number_input("MDVP:Shimmer")
    with col1:
        MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)")
    with col2:
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3")
    with col3:
        Shimmer_APQ5 = st.number_input("Shimmer:APQ5")
    with col1:
        MDVP_APQ = st.number_input("MDVP:APQ")
    with col2:
        Shimmer_DDA = st.number_input("Shimmer:DDA")
    with col3:
        NHR = st.number_input("NHR")
    with col1:
        HNR = st.number_input("HNR")
    with col2:
        RPDE = st.number_input("RPDE")
    with col3:
        DFA = st.number_input("DFA")
    with col1:
        spread1 = st.number_input("spread1")
    with col2:
        spread2 = st.number_input("spread2")
    with col3:
        D2 = st.number_input("D2")
    with col1:
        PPE = st.number_input("PPE")

    # Code for prediction
    parkinson_dig = ''

    # Button
    if st.button("Parkinson test result"):
        parkinson_prediction = parkinson_model.predict(
            [[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
        )

        # Determine the result
        if parkinson_prediction[0] == 1:
            result = "Parkinson's Disease"
            parkinson_dig = "We are really sorry to say but it seems like you have Parkinson's disease."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            result = "No Parkinson's Disease"
            parkinson_dig = "Congratulations, you don't have Parkinson's disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        # Save prediction history
        save_prediction_history(st.session_state["username"], "Parkinson's Disease", result)

        # Display result
        st.success(f"{name}, {parkinson_dig}")



# Load the dataset
lung_cancer_data = pd.read_csv('data/lung_cancer.csv')

# Convert 'M' to 0 and 'F' to 1 in the 'GENDER' column
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

# Lung Cancer prediction page
if selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")
    image = Image.open('download (2).jpg')
    st.image(image, caption='Lung Cancer Prediction')

    # Columns
    # No inputs from the user
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())
    with col2:
        age = st.number_input("Age")
    with col3:
        smoking = st.selectbox("Smoking:", ['NO', 'YES'])
    with col1:
        yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])

    with col2:
        anxiety = st.selectbox("Anxiety:", ['NO', 'YES'])
    with col3:
        peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
    with col1:
        chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])

    with col2:
        fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
    with col3:
        allergy = st.selectbox("Allergy:", ['NO', 'YES'])
    with col1:
        wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])

    with col2:
        alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
    with col3:
        coughing = st.selectbox("Coughing:", ['NO', 'YES'])
    with col1:
        shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])

    with col2:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
    with col3:
        chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

    # Code for prediction
    cancer_result = ''

    # Button
    if st.button("Predict Lung Cancer"):
        # Create a DataFrame with user inputs
        user_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue],
            'ALLERGY': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })

        # Map string values to numeric
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

        # Strip leading and trailing whitespaces from column names
        user_data.columns = user_data.columns.str.strip()

        # Convert columns to numeric where necessary
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Perform prediction
        cancer_prediction = lung_cancer_model.predict(user_data)

        # Display result
        if cancer_prediction[0] == 'YES':
            cancer_result = "The model predicts that there is a risk of Lung Cancer."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            cancer_result = "The model predicts no significant risk of Lung Cancer."
            image = Image.open('negative.jpg')
            st.image(image, caption='')
         
        save_prediction_history(st.session_state["username"], "Lung cancer", result)
        st.success(name + ', ' + cancer_result)


if selected == 'Liver Prediction':
    st.title("Liver Disease Prediction")
    image = Image.open('download (6).jpg')
    st.image(image, caption='Liver disease prediction')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Sex = st.selectbox("Gender", ["Male", "Female"])
        Sex = 1 if Sex == "Male" else 0
    with col2:
        Age = st.number_input("Age")
    with col3:
        Total_Bilirubin = st.number_input("Total Bilirubin")
    with col1:
        Direct_Bilirubin = st.number_input("Direct Bilirubin")
    with col2:
        Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase")
    with col3:
        Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase")
    with col1:
        Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase")
    with col2:
        Total_Protiens = st.number_input("Total Proteins")
    with col3:
        Albumin = st.number_input("Albumin")
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio")

    # Code for prediction
    liver_dig = ''

    # Button
    if st.button("Liver test result"):
        liver_prediction = liver_model.predict(
            [[Sex, Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]]
        )

        # Determine the result
        if liver_prediction[0] == 1:
            result = "Liver Disease"
            liver_dig = "We are really sorry to say but it seems like you have Liver Disease."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            result = "No Liver Disease"
            liver_dig = "Congratulations, you don't have Liver Disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        # Save prediction history
        save_prediction_history(st.session_state["username"], "Liver Disease", result)

        # Display result
        st.success(f"{name}, {liver_dig}")

    



if selected == 'Hepatitis Prediction':
    st.title("Hepatitis Prediction")
    image = Image.open('download (1).jpg')
    st.image(image, caption='Hepatitis Prediction')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input("Age")
    with col2:
        Sex = st.selectbox("Gender", ["Male", "Female"])
        Sex = 1 if Sex == "Male" else 2
    with col3:
        Total_Bilirubin = st.number_input("Total Bilirubin")
    with col1:
        Direct_Bilirubin = st.number_input("Direct Bilirubin")
    with col2:
        Alkaline_Phosphatase = st.number_input("Alkaline Phosphatase")
    with col3:
        Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase")
    with col1:
        Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase")
    with col2:
        Total_Proteins = st.number_input("Total Proteins")
    with col3:
        Albumin = st.number_input("Albumin")
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio")
    with col2:
        GGT = st.number_input("GGT")  # Add this feature
    with col3:
        PROT = st.number_input("PROT")  # Add this feature

    # Code for prediction
    hepatitis_result = ''

    # Button
    if st.button("Predict Hepatitis"):
        # Create a DataFrame with user inputs (ensure all 12 features are included)
        user_data = pd.DataFrame({
            'Age': [Age],
            'Sex': [Sex],
            'Total_Bilirubin': [Total_Bilirubin],
            'Direct_Bilirubin': [Direct_Bilirubin],
            'Alkaline_Phosphatase': [Alkaline_Phosphatase],
            'Alamine_Aminotransferase': [Alamine_Aminotransferase],
            'Aspartate_Aminotransferase': [Aspartate_Aminotransferase],
            'Total_Proteins': [Total_Proteins],
            'Albumin': [Albumin],
            'Albumin_and_Globulin_Ratio': [Albumin_and_Globulin_Ratio],
            'GGT': [GGT],  # Add this feature
            'PROT': [PROT]  # Add this feature
        })

        # Perform prediction
        hepatitis_prediction = hepatitis_model.predict(user_data)

        # Determine the result
        if hepatitis_prediction[0] == 1:
            result = "Hepatitis"
            hepatitis_result = "We are really sorry to say but it seems like you have Hepatitis."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            result = "No Hepatitis"
            hepatitis_result = "Congratulations, you do not have Hepatitis."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        # Save prediction history
        save_prediction_history(st.session_state["username"], "Hepatitis", result)

        # Display result
        st.success(f"{name}, {hepatitis_result}")





# jaundice prediction page
if selected == 'Jaundice prediction':  # pagetitle
    st.title("Jaundice disease prediction")
    image = Image.open('j.jpg')
    st.image(image, caption='Jaundice disease prediction')
    # columns
    # no inputs from the user
# st.write(info.astype(int).info())
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Entre your age   ") # 2 
    with col2:
        Sex=0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x])
        if value == "male":
            Sex = 0
        elif value == "female":
            Sex = 1
    with col3:
        Total_Bilirubin = st.number_input("Entre your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Entre your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Entre your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Entre your Alamine_Aminotransferase") # 6
    with col1:
        Total_Protiens = st.number_input("Entre your Total_Protiens")# 8
    with col2:
        Albumin = st.number_input("Entre your Albumin") # 9 
    # code for prediction
    jaundice_dig = ''

    # button
    if st.button("Jaundice test result"):
        jaundice_prediction=[[]]
        jaundice_prediction = jaundice_model.predict([[age,Sex,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Total_Protiens,Albumin]])

        # after the prediction is done if the value in the list at index is 0 is 1 then the person is diabetic
        if jaundice_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            jaundice_dig = "we are really sorry to say but it seems like you have Jaundice."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            jaundice_dig = "Congratulation , You don't have Jaundice."

        save_prediction_history(st.session_state["username"], "jaundice_result", result)    
        st.success(name+' , ' + jaundice_dig)












from sklearn.preprocessing import LabelEncoder
import joblib


# Chronic Kidney Disease Prediction Page
if selected == 'Chronic Kidney Prediction':
    st.title("Chronic Kidney Disease Prediction")
    image = Image.open('kidney.jpg')
    st.image(image, caption='Chronic Kidney Disease Prediction')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input("Age")
    with col2:
        Blood_Pressure = st.number_input("Blood Pressure")
    with col3:
        Specific_Gravity = st.number_input("Specific Gravity")
    with col1:
        Albumin = st.number_input("Albumin")
    with col2:
        Sugar = st.number_input("Sugar")
    with col3:
        Red_Blood_Cells = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"])
        Red_Blood_Cells = 1 if Red_Blood_Cells == "Normal" else 0
    with col1:
        Pus_Cells = st.selectbox("Pus Cells", ["Normal", "Abnormal"])
        Pus_Cells = 1 if Pus_Cells == "Normal" else 0
    with col2:
        Pus_Cell_Clumps = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"])
        Pus_Cell_Clumps = 1 if Pus_Cell_Clumps == "Present" else 0
    with col3:
        Bacteria = st.selectbox("Bacteria", ["Present", "Not Present"])
        Bacteria = 1 if Bacteria == "Present" else 0
    with col1:
        Blood_Glucose_Random = st.number_input("Blood Glucose Random")
    with col2:
        Blood_Urea = st.number_input("Blood Urea")
    with col3:
        Serum_Creatinine = st.number_input("Serum Creatinine")
    with col1:
        Sodium = st.number_input("Sodium")
    with col2:
        Potassium = st.number_input("Potassium")
    with col3:
        Hemoglobin = st.number_input("Hemoglobin")
    with col1:
        Packed_Cell_Volume = st.number_input("Packed Cell Volume")
    with col2:
        White_Blood_Cell_Count = st.number_input("White Blood Cell Count")
    with col3:
        Red_Blood_Cell_Count = st.number_input("Red Blood Cell Count")
    with col1:
        Hypertension = st.selectbox("Hypertension", ["Yes", "No"])
        Hypertension = 1 if Hypertension == "Yes" else 0
    with col2:
        Diabetes_Mellitus = st.selectbox("Diabetes Mellitus", ["Yes", "No"])
        Diabetes_Mellitus = 1 if Diabetes_Mellitus == "Yes" else 0
    with col3:
        Coronary_Artery_Disease = st.selectbox("Coronary Artery Disease", ["Yes", "No"])
        Coronary_Artery_Disease = 1 if Coronary_Artery_Disease == "Yes" else 0
    with col1:
        Appetite = st.selectbox("Appetite", ["Good", "Poor"])
        Appetite = 1 if Appetite == "Good" else 0
    with col2:
        Pedal_Edema = st.selectbox("Pedal Edema", ["Yes", "No"])
        Pedal_Edema = 1 if Pedal_Edema == "Yes" else 0
    with col3:
        Anemia = st.selectbox("Anemia", ["Yes", "No"])
        Anemia = 1 if Anemia == "Yes" else 0

    # Code for prediction
    kidney_result = ''

    # Button
    if st.button("Predict Chronic Kidney Disease"):
        kidney_prediction = chronic_disease_model.predict(
            [[Age, Blood_Pressure, Specific_Gravity, Albumin, Sugar, Red_Blood_Cells, Pus_Cells, Pus_Cell_Clumps, Bacteria, Blood_Glucose_Random, Blood_Urea, Serum_Creatinine, Sodium, Potassium, Hemoglobin, Packed_Cell_Volume, White_Blood_Cell_Count, Red_Blood_Cell_Count, Hypertension, Diabetes_Mellitus, Coronary_Artery_Disease, Appetite, Pedal_Edema, Anemia]]
        )

        # Determine the result
        if kidney_prediction[0] == 1:
            result = "Chronic Kidney Disease"
            kidney_result = "We are really sorry to say but it seems like you have Chronic Kidney Disease."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            result = "No Chronic Kidney Disease"
            kidney_result = "Congratulations, you don't have Chronic Kidney Disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        # Save prediction history
        save_prediction_history(st.session_state["username"], "Chronic Kidney Disease", result)

        # Display result
        st.success(f"{name}, {kidney_result}")


# Breast Cancer Prediction Page
if selected == 'Breast Cancer Prediction':
    st.title("Breast Cancer Prediction")
    image = Image.open('download.jpg')
    st.image(image, caption='Breast Cancer Prediction')
    
    # Input fields
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        Radius_Mean = st.number_input("Radius Mean")
    with col2:
        Texture_Mean = st.number_input("Texture Mean")
    with col3:
        Perimeter_Mean = st.number_input("Perimeter Mean")
    with col1:
        Area_Mean = st.number_input("Area Mean")
    with col2:
        Smoothness_Mean = st.number_input("Smoothness Mean")
    with col3:
        Compactness_Mean = st.number_input("Compactness Mean")
    with col1:
        Concavity_Mean = st.number_input("Concavity Mean")
    with col2:
        Concave_Points_Mean = st.number_input("Concave Points Mean")
    with col3:
        Symmetry_Mean = st.number_input("Symmetry Mean")
    with col1:
        Fractal_Dimension_Mean = st.number_input("Fractal Dimension Mean")
    with col2:
        Radius_SE = st.number_input("Radius SE")
    with col3:
        Texture_SE = st.number_input("Texture SE")
    with col1:
        Perimeter_SE = st.number_input("Perimeter SE")
    with col2:
        Area_SE = st.number_input("Area SE")
    with col3:
        Smoothness_SE = st.number_input("Smoothness SE")
    with col1:
        Compactness_SE = st.number_input("Compactness SE")
    with col2:
        Concavity_SE = st.number_input("Concavity SE")
    with col3:
        Concave_Points_SE = st.number_input("Concave Points SE")
    with col1:
        Symmetry_SE = st.number_input("Symmetry SE")
    with col2:
        Fractal_Dimension_SE = st.number_input("Fractal Dimension SE")
    with col3:
        Radius_Worst = st.number_input("Radius Worst")
    with col1:
        Texture_Worst = st.number_input("Texture Worst")
    with col2:
        Perimeter_Worst = st.number_input("Perimeter Worst")
    with col3:
        Area_Worst = st.number_input("Area Worst")
    with col1:
        Smoothness_Worst = st.number_input("Smoothness Worst")
    with col2:
        Compactness_Worst = st.number_input("Compactness Worst")
    with col3:
        Concavity_Worst = st.number_input("Concavity Worst")
    with col1:
        Concave_Points_Worst = st.number_input("Concave Points Worst")
    with col2:
        Symmetry_Worst = st.number_input("Symmetry Worst")
    with col3:
        Fractal_Dimension_Worst = st.number_input("Fractal Dimension Worst")

    # Code for prediction
    breast_cancer_result = ''

    # Button
    if st.button("Predict Breast Cancer"):
        # Create a DataFrame with user inputs
        user_input = pd.DataFrame({
            'radius_mean': [Radius_Mean],
            'texture_mean': [Texture_Mean],
            'perimeter_mean': [Perimeter_Mean],
            'area_mean': [Area_Mean],
            'smoothness_mean': [Smoothness_Mean],
            'compactness_mean': [Compactness_Mean],
            'concavity_mean': [Concavity_Mean],
            'concave points_mean': [Concave_Points_Mean],
            'symmetry_mean': [Symmetry_Mean],
            'fractal_dimension_mean': [Fractal_Dimension_Mean],
            'radius_se': [Radius_SE],
            'texture_se': [Texture_SE],
            'perimeter_se': [Perimeter_SE],
            'area_se': [Area_SE],
            'smoothness_se': [Smoothness_SE],
            'compactness_se': [Compactness_SE],
            'concavity_se': [Concavity_SE],
            'concave points_se': [Concave_Points_SE],
            'symmetry_se': [Symmetry_SE],
            'fractal_dimension_se': [Fractal_Dimension_SE],
            'radius_worst': [Radius_Worst],
            'texture_worst': [Texture_Worst],
            'perimeter_worst': [Perimeter_Worst],
            'area_worst': [Area_Worst],
            'smoothness_worst': [Smoothness_Worst],
            'compactness_worst': [Compactness_Worst],
            'concavity_worst': [Concavity_Worst],
            'concave points_worst': [Concave_Points_Worst],
            'symmetry_worst': [Symmetry_Worst],
            'fractal_dimension_worst': [Fractal_Dimension_Worst],
        })

        # Perform prediction
        breast_cancer_prediction = breast_cancer_model.predict(user_input)

        # Determine the result
        if breast_cancer_prediction[0] == 1:
            result = "Breast Cancer"
            breast_cancer_result = "The model predicts that you have Breast Cancer."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            result = "No Breast Cancer"
            breast_cancer_result = "The model predicts that you don't have Breast Cancer."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        # Save prediction history
        save_prediction_history(st.session_state["username"], "Breast Cancer", result)

        # Display result
        st.success(breast_cancer_result)

        
# ==================================================
# Prediction History
# ==================================================
if selected == 'Prediction History':
    st.title("Prediction History")
    st.write("### View your past predictions and results.")

    # Fetch prediction history for the logged-in user
    history = get_prediction_history(st.session_state["username"])
    
    if history:
        st.write("#### Your Prediction History:")
        for entry in history:
            st.write(f"""
                - **Disease Type:** {entry[2]}
                - **Result:** {entry[3]}
                - **Date:** {entry[4]}
            """)
    else:
        st.write("No prediction history found. Start making predictions to see your history here!")

# ==================================================
# AI Chatbot
# ==================================================
if selected == 'AI Chatbot':
    st.title("AI Chatbot for Symptom Analysis")
    st.write("### Describe your symptoms and get instant health advice.")

    user_input = st.text_area("Enter your symptoms or health concerns:")
    
    if st.button("Get Advice"):
        if user_input.strip():
            # Simulate AI response (replace with actual AI integration)
            response = f"Based on your symptoms: {user_input}, it is recommended to consult a doctor for further evaluation."
            st.write("### AI Response:")
            st.write(response)
        else:
            st.error("Please describe your symptoms to get advice.")

# ==================================================
# Book Consultation
# ==================================================
if selected == 'Book Consultation':
    st.title("Book a Doctor Consultation")
    st.write("### Find and book appointments with certified doctors.")

    # List of available doctors
    doctors = [
        {"name": "Dr. Smith", "specialization": "Cardiologist", "availability": "Mon-Fri, 9 AM - 5 PM"},
        {"name": "Dr. Johnson", "specialization": "Dermatologist", "availability": "Tue-Thu, 10 AM - 4 PM"},
        {"name": "Dr. Lee", "specialization": "General Physician", "availability": "Mon-Sat, 8 AM - 6 PM"},
    ]

    st.write("#### Available Doctors:")
    for doctor in doctors:
        st.write(f"""
            - **Name:** {doctor['name']}
            - **Specialization:** {doctor['specialization']}
            - **Availability:** {doctor['availability']}
        """)

    selected_doctor = st.selectbox("Select a Doctor", [doc["name"] for doc in doctors])
    appointment_date = st.date_input("Select Appointment Date")
    appointment_time = st.time_input("Select Appointment Time")

    if st.button("Book Appointment"):
        st.success(f"Appointment booked with {selected_doctor} on {appointment_date} at {appointment_time}!")

# ==================================================
# Health Tips
# ==================================================
if selected == 'Health Tips':
    st.title("Health Tips and Articles")
    st.write("### Stay healthy with these tips and articles.")

    st.write("#### Daily Health Tips:")
    tips = [
        "Drink at least 8 glasses of water daily.",
        "Exercise for 30 minutes every day.",
        "Eat a balanced diet rich in fruits and vegetables.",
        "Get 7-8 hours of sleep every night.",
        "Avoid smoking and limit alcohol consumption.",
    ]
    for tip in tips:
        st.write(f"- {tip}")

    st.write("#### Featured Articles:")
    articles = [
        {"title": "10 Ways to Boost Your Immune System", "link": "#"},
        {"title": "Understanding Heart Health", "link": "#"},
        {"title": "The Importance of Mental Health", "link": "#"},
    ]
    for article in articles:
        st.write(f"- [{article['title']}]({article['link']})")

# ==================================================
# Health Trends
# ==================================================
if selected == 'Health Trends':
    st.title("Health Trends Visualization")
    st.write("### Track your health trends over time.")

    # Sample data for visualization
    data = pd.DataFrame({
        "Date": pd.date_range(start="2023-01-01", periods=30),
        "Weight": np.random.randint(60, 80, size=30),
        "Blood Pressure": np.random.randint(110, 140, size=30),
    })

    st.write("#### Weight Over Time")
    st.line_chart(data.set_index("Date")["Weight"])

    st.write("#### Blood Pressure Over Time")
    st.line_chart(data.set_index("Date")["Blood Pressure"])

# ==================================================
# Feedback
# ==================================================
if selected == 'Feedback':
    st.title("Provide Feedback")
    st.write("### We value your feedback! Share your thoughts with us.")

    feedback = st.text_area("Enter your feedback:")
    rating = st.slider("Rate your experience (1 = Poor, 5 = Excellent)", 1, 5)

    if st.button("Submit Feedback"):
        if feedback.strip():
            st.success("Thank you for your feedback! We appreciate your input.")
        else:
            st.error("Please provide feedback before submitting.")

# ==================================================
# Achievements
# ==================================================
if selected == 'Achievements':
    st.title("Your Achievements")
    st.write("### Earn badges for your activity on HealthGuard Pro.")

    # Sample badges
    badges = [
        {"name": "First Prediction", "earned": True},
        {"name": "5 Predictions", "earned": False},
        {"name": "Health Enthusiast", "earned": True},
        {"name": "Feedback Contributor", "earned": False},
    ]

    st.write("#### Badges Earned:")
    for badge in badges:
        if badge["earned"]:
            st.write(f"- {badge['name']} ✅")
        else:
            st.write(f"- {badge['name']} ❌ (Keep using the app to earn this badge!)")