import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Penguin Prediction App

This app predicts the **Palmer Penguin** species!

Data is obtained from the [palmerpenguins](https://github.com/allisonhorst/palmerpenguins) library in R by Allison Horst.
""")

st.sidebar.header("User Input Features")
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/penguins_example.csv)
""")

# Collect user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox("Island", ("Biscoe", "Dream", "Torgersen"))
        sex = st.sidebar.selectbox("Sex", ("male", "female"))
        bill_length_mm = st.sidebar.slider("Bill length (mm)", 32.1, 59.6, 43.9)
        bill_dept_mm = st.sidebar.slider("Bill depth (mm)", 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider("Body mass (g)", 2700.0, 6300.0, 4207.0)
        data = {"island": island,
                "sex": sex,
                "bill_length_mm": bill_length_mm,
                "bill_depth_mm": bill_dept_mm,
                "flipper_length_mm": flipper_length_mm,
                "body_mass_g": body_mass_g,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combine user input features with entire penguins dataset
# This will be useful for the encoding phase
# If we don't concat the input with the raw dataset, sex will have only 1 value
# --> cannot split into 2 categorical columns
# Same for Island.
# And the below block of code will not work correctly
penguins_raw = pd.read_csv("./Penguin_Classification/penguins_cleaned.csv")
penguins = penguins_raw.drop(columns=["species"])
df = pd.concat([input_df, penguins], axis=0)

# Encoding of ordinal features
encode = ["sex", "island"]
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]     # Select only the first row (the user input data)

# Displays the user input features
st.write("-" * 50)
st.subheader('User Input parameters')
if uploaded_file is None:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
st.write(df.to_markdown(index=False))

# Reads in saved classification model
load_classifier = pickle.load(open("./Penguin_Classification/penguins_classifier.pkl", "rb"))

# Apply model to make predictions
prediction = load_classifier.predict(df)
prediction_proba = load_classifier.predict_proba(df)

# Display results
st.write("-" * 50)
st.subheader("Prediction")
penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
st.write(penguins_species[prediction[0]])


st.write("-" * 50)
st.subheader("Prediction Probability")
prediction_proba = {cls_name: prediction_proba[0][ind] for (ind, cls_name) in enumerate(penguins_species)}
prediction_proba = pd.DataFrame(prediction_proba, index=[0])
st.write(prediction_proba.to_markdown(index=False))
