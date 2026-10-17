# Import Path to create file paths that work relative to this project.
from pathlib import Path

# Import joblib to load the trained machine-learning model.
import joblib
# Import pandas for reading the dataset and creating prediction input data.
import pandas as pd
# Import Streamlit to build the interactive web application.
import streamlit as st


# Define the root directory of the project using the location of this Python file.
PROJECT_ROOT = Path(__file__).resolve().parent
# Define the path where the trained model is stored.
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
# Define the path of the cacao dataset used to populate the application inputs.
DATA_PATH = PROJECT_ROOT / "data" / "flavors_of_cacao.csv"

# Configure the Streamlit page title, icon, and centered layout.
st.set_page_config(page_title="Cacao Rating Predictor", page_icon="🍫", layout="centered")

# Add custom CSS styling to improve the visual appearance of the Streamlit application.
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #fffaf3, #f3e3cf); }
.block-container { max-width: 900px; }
h1, h2, h3 { color: #4b2e20; }
.result-card { padding: 1.2rem; border-radius: 14px; background: #4b2e20; color: white;
               text-align: center; font-size: 1.5rem; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# Display the main application title.
st.title("🍫 Chocolate Bar Rating Predictor")
# Explain to the user what the application predicts.
st.write("Enter the chocolate details to estimate its expert rating on a 1–5 scale.")

# Check whether the trained model file exists before continuing.
if not MODEL_PATH.exists():
    st.error("Model not found. Open the VS Code terminal and run: `python src/train.py`")
    st.stop()

# Read the cacao dataset so its values can be used in the input controls.
raw = pd.read_csv(DATA_PATH)
# Extract unique company/maker names from the first dataset column and sort them.
companies = sorted(raw.iloc[:, 0].dropna().astype(str).str.strip().unique())
# Extract and sort the available company locations.
locations = sorted(raw["Company\nLocation"].dropna().astype(str).str.strip().unique())
# Extract and sort non-empty bean types.
bean_types = sorted(x for x in raw["Bean\nType"].dropna().astype(str).str.strip().unique() if x)
# Extract and sort non-empty broad bean origins.
origins = sorted(x for x in raw["Broad Bean\nOrigin"].dropna().astype(str).str.strip().unique() if x)

# Load the saved machine-learning model pipeline.
model = joblib.load(MODEL_PATH)
# Create two Streamlit columns for arranging the input controls.
left, right = st.columns(2)
# Place the first group of input controls in the left column.
with left:
# Let the user select the chocolate company or maker.
    company = st.selectbox("Company / Maker", companies)
# Let the user enter the name of the chocolate bar.
    bar_name = st.text_input("Bar Name", "New Chocolate Bar")
# Let the user select the company location.
    company_location = st.selectbox("Company Location", locations)
# Let the user select the broad bean origin.
    bean_origin = st.selectbox("Broad Bean Origin", origins)
# Place the second group of input controls in the right column.
with right:
# Let the user select the cocoa percentage using a slider.
    cocoa_percent = st.slider("Cocoa Percentage", 40, 100, 70)
# Let the user enter/select the review year.
    review_year = st.number_input("Review Year", 2006, 2030, 2017)
# Let the user enter/select the reference number.
    ref = st.number_input("Reference Number", 1, 5000, 2000)
# Let the user select the bean type.
    bean_type = st.selectbox("Bean Type", bean_types)

# Run the prediction process when the user clicks the Predict Rating button.
if st.button("Predict Rating", type="primary", use_container_width=True):
# Create a one-row DataFrame containing the user's input values.
    sample = pd.DataFrame([{
        "company": company,
        "bar_name": bar_name,
        "ref": ref,
        "review_year": review_year,
        "cocoa_percent": cocoa_percent,
        "company_location": company_location,
        "bean_type": bean_type,
        "bean_origin": bean_origin,
    }])
# Use the loaded model to predict the rating for the submitted chocolate details.
    prediction = float(model.predict(sample)[0])
# Keep the predicted rating within the intended 1-to-5 rating range.
    prediction = max(1.0, min(5.0, prediction))
# Display the prediction using the custom result-card styling.
    st.markdown(
        f'<div class="result-card">Predicted Rating: {prediction:.2f} / 5.00</div>',
        unsafe_allow_html=True,
    )
# Show a short note explaining the basis of the prediction.
    st.caption("This is an ML estimate based on historical chocolate reviews.")

# Provide additional information about the models used in the project.
with st.expander("About this project"):
# Explain the model comparison and selection used by the application.
    st.write("The app compares Linear, Ridge, Random Forest, Gradient Boosting, and tuned Random Forest regression models. The best test-RMSE model is saved for prediction.")

