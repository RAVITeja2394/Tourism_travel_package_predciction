import os
import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load the model committed by the pipeline
model_path = os.path.join(os.path.dirname(__file__), "best_model_tourism_package_prediction_v1.joblib")

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

try:
    model = load_pipeline(model_path)
except Exception as e:
    st.error(f"Failed to load model file. Error: {e}")
    st.stop()

st.title("Tourism Package Conversion Prediction App")
st.write("Enter raw customer information below. The app will automatically compute advanced metrics for the model.")

st.header("Customer Demographics")
col1, col2 = st.columns(2)

with col1:
    raw_age = st.number_input("Customer Age", min_value=1, max_value=120, value=30, step=1)
    CityTier = st.selectbox("City Tier Code", ["1", "2", "3"], index=0)
    Occupation = st.selectbox("Occupation Category", ["Salaried", "Small Business", "Large Business", "Free Lancer"], index=0)

with col2:
    MonthlyIncome = st.number_input("Monthly Income (INR)", min_value=0.0, max_value=500000.0, value=25000.0, step=500.0)
    Life_Stage = st.selectbox("Life Stage Profile", ["Single", "Married", "Divorced", "Unmarried"], index=1)
    Passport = st.selectbox("Possesses Valid Passport?", [1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)", index=1)

st.header("Sales Interaction Details")
col3, col4 = st.columns(2)

with col3:
    TypeofContact = st.selectbox("Primary Lead Contact Channel", ["Self Enquiry", "Company Invited"], index=0)
    ProductPitched = st.selectbox("Holiday Package Type Pitched", ["Deluxe", "Standard", "Basic", "Super Deluxe", "King"], index=2)
    DurationOfPitch = st.number_input("Pitch Duration (Minutes)", min_value=1.0, max_value=120.0, value=15.0, step=1.0)

with col4:
    NumberOfFollowups = st.number_input("Total Sales Follow-up Sessions", min_value=1.0, max_value=20.0, value=4.0, step=1.0)
    PitchSatisfactionScore = st.selectbox("Pitch Customer Feedback Score", ["1", "2", "3", "4", "5"], index=2)
    NumberOfTrips = st.number_input("Previous Historic Trips Booked", min_value=0.0, max_value=30.0, value=2.0, step=1.0)
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", ["3", "4", "5"], index=0)
    Total_Group_Size = st.number_input("Total Accompanied Party Size", min_value=1, max_value=20, value=2, step=1)

# =====================================================================
# 3. AUTOMATED BEHIND-THE-SCENES FEATURE ENGINEERING
# =====================================================================

# A. Calculate Age Bucket
if raw_age <= 18:
    Age_bucket = '0-18'
elif raw_age <= 25:
    Age_bucket = '19-25'
elif raw_age <= 35:
    Age_bucket = '26-35'
elif raw_age <= 50:
    Age_bucket = '36-50'
else:
    Age_bucket = '51+'

# B. Calculate Interaction Efficiency (Handle division by zero safely just in case)
Interaction_Efficiency = DurationOfPitch / NumberOfFollowups if NumberOfFollowups > 0 else 0.0

# C. Calculate Is_Young_Executive
Is_Young_Executive = 1 if (raw_age <= 35 and Occupation == "Salaried") else 0

# D. Calculate Pitch Alignment (Example rule: high satisfaction score on standard/basic packages)
Pitch_Alignment = 1 if (int(PitchSatisfactionScore) >= 4 and ProductPitched in ["Basic", "Standard"]) else 0


# Constructing input DataFrame matching pipeline expectation exactly
input_data = pd.DataFrame([{
    "DurationOfPitch": DurationOfPitch,
    "MonthlyIncome": MonthlyIncome,
    "NumberOfFollowups": NumberOfFollowups,
    "NumberOfTrips": NumberOfTrips,
    "Interaction_Efficiency": Interaction_Efficiency, 
    "TypeofContact": TypeofContact,
    "ProductPitched": ProductPitched,
    "Occupation": Occupation,
    "Life_Stage": Life_Stage,
    "CityTier": CityTier,
    "PreferredPropertyStar": PreferredPropertyStar,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "Age_bucket": Age_bucket,                         
    "Passport": Passport,
    "Total_Group_Size": Total_Group_Size,
    "Is_Young_Executive": Is_Young_Executive,         
    "Pitch_Alignment": Pitch_Alignment                 
}])

# Enforce strict clean data typings matching prep.py assertions

categorical_cols = ['TypeofContact', 'ProductPitched', 'Occupation', 'Life_Stage', 
                    'CityTier', 'PreferredPropertyStar', 'PitchSatisfactionScore', 'Age_bucket']
for col in categorical_cols:
    input_data[col] = input_data[col].astype(str)

numeric_cols = ['DurationOfPitch', 'MonthlyIncome', 'NumberOfFollowups', 'NumberOfTrips', 'Interaction_Efficiency']
for col in numeric_cols:
    input_data[col] = input_data[col].astype(float)


feature_order = [
    'DurationOfPitch', 'MonthlyIncome', 'NumberOfFollowups', 'NumberOfTrips', 'Interaction_Efficiency',
    'TypeofContact', 'ProductPitched', 'Occupation', 'Life_Stage', 'Age_bucket',
    'CityTier', 'PreferredPropertyStar', 'PitchSatisfactionScore', 'Passport', 
    'Total_Group_Size', 'Is_Young_Executive', 'Pitch_Alignment'
]
input_data = input_data[feature_order] 

# Execute prediction step safely
if st.button("Evaluate Package Purchase Likelihood"):
    probabilities = model.predict_proba(input_data)


# Execute prediction step
if st.button("Evaluate Package Purchase Likelihood"):
    probabilities = model.predict_proba(input_data)
    purchase_probability = probabilities[0][1]
    
    classification_threshold = 0.45
    prediction = 1 if purchase_probability >= classification_threshold else 0
    
    st.subheader("Analysis & Outcome Breakdown:")
    
    if prediction == 1:
        st.success(f"### **Prediction: High Conversion Potential (Target Class 1)**")
        st.metric(label="Calculated Purchase Propensity", value=f"{purchase_probability * 100:.2f}%", delta="Exceeds 45% Threshold")
        st.balloons()
    else:
        st.warning(f"### **Prediction: Low Conversion Potential (Target Class 0)**")
        st.metric(label="Calculated Purchase Propensity", value=f"{purchase_probability * 100:.2f}%", delta=f"-{(classification_threshold - purchase_probability)*100:.2f}% to threshold", delta_color="inverse")
