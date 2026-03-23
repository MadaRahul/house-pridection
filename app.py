import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("house_price_model.pkl")

# Load training data to get all columns
train = pd.read_csv("train.csv")
all_columns = [col for col in train.columns if col not in ['SalePrice', 'Id']]

# Title
st.title("House Price Prediction App")

# Input features (simplified - using key features)
st.header("Enter House Features")

# Numerical features
lot_area = st.number_input("Lot Area (sq ft)", min_value=0, value=10000)
year_built = st.number_input("Year Built", min_value=1800, max_value=2023, value=2000)
total_bsmt_sf = st.number_input("Total Basement SF", min_value=0, value=1000)
gr_liv_area = st.number_input("Above Ground Living Area SF", min_value=0, value=1500)
full_bath = st.number_input("Full Bathrooms", min_value=0, value=2)
bedroom_abv_gr = st.number_input("Bedrooms Above Ground", min_value=0, value=3)
garage_cars = st.number_input("Garage Cars", min_value=0, value=2)

# Categorical features (simplified)
overall_qual = st.selectbox("Overall Quality", [1,2,3,4,5,6,7,8,9,10], index=5)
neighborhood = st.selectbox("Neighborhood", ["CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "SawyerW", "IDOTRR", "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"], index=0)
house_style = st.selectbox("House Style", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"], index=0)

# Create input dataframe with all columns
input_data = pd.DataFrame(columns=all_columns)

# Set defaults
for col in all_columns:
    if train[col].dtype in ['int64', 'float64']:
        input_data.loc[0, col] = 0  # or train[col].median()
    else:
        input_data.loc[0, col] = 'NA'  # common for missing

# Set user inputs
input_data.loc[0, 'LotArea'] = lot_area
input_data.loc[0, 'YearBuilt'] = year_built
input_data.loc[0, 'TotalBsmtSF'] = total_bsmt_sf
input_data.loc[0, 'GrLivArea'] = gr_liv_area
input_data.loc[0, 'FullBath'] = full_bath
input_data.loc[0, 'BedroomAbvGr'] = bedroom_abv_gr
input_data.loc[0, 'GarageCars'] = garage_cars
input_data.loc[0, 'OverallQual'] = overall_qual
input_data.loc[0, 'Neighborhood'] = neighborhood
input_data.loc[0, 'HouseStyle'] = house_style

# Predict
if st.button("Predict Price"):
    prediction_log = model.predict(input_data)
    prediction = np.expm1(prediction_log)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")

# Note
st.write("Note: This is a simplified app using key features. Full model uses all 79 features.")