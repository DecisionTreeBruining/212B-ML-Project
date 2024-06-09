from shiny import App, render, ui, reactive
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
#from asgmnt_2_tools import lazy_read_parquet

file_path = "../../Data/heart_2022_with_nans.csv"
df = pd.read_csv(file_path)

# Load the model
#model_path = "../../Data/GoogleDrive/MLP_Results/best_model_best_params-Under_Sample_1:1_threshold_20.pkl"
model_path = "../models/best_model.pkl"
with open(model_path, 'rb') as file:
    best_model = pickle.load(file)

with open('../models/preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

with open('../models/standard_scaler.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', standard_scaler),
    ('model', best_model)
])
  

# UI section starts from here 
app_ui = ui.page_fluid(
    ui.markdown(
        """
        ## Heart Disease Prediction Model
        """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_select("State", "Which state do you live in?", 
                            {0: 'Alabama', 1: 'Alaska', 2: 'Arizona', 3: 'Arkansas',
                             4: 'California', 5: 'Colorado', 6: 'Connecticut', 7: 'Delaware',
                             8: 'District of Columbia', 9: 'Florida', 10: 'Georgia', 11: 'Guam', 
                             12: 'Hawaii', 13: 'Idaho', 14: 'Illinois', 15: 'Indiana', 16: 'Iowa',
                             17: 'Kansas', 18: 'Kentucky', 19: 'Louisiana', 20: 'Maine', 21: 'Maryland',
                             22: 'Massachusetts', 23: 'Michigan', 24: 'Minnesota', 25: 'Mississippi', 
                             26: 'Missouri', 27: 'Montana', 28: 'Nebraska', 29: 'Nevada', 
                             30: 'New Hampshire', 31: 'New Jersey', 32: 'New Mexico', 33: 'New York', 
                             34: 'North Carolina', 35: 'North Dakota', 36: 'Ohio', 37: 'Oklahoma', 
                             38: 'Oregon', 39: 'Pennsylvania', 40: 'Puerto Rico', 41: 'Rhode Island', 
                             42: 'South Carolina', 43: 'South Dakota', 44: 'Tennessee', 45: 'Texas', 
                             46: 'Utah', 47: 'Vermont', 48: 'Virgin Islands', 49: 'Virginia', 
                             50: 'Washington', 51: 'West Virginia', 52: 'Wisconsin', 53: 'Wyoming'}),
            ui.input_select("Sex", "Sex", {0: "Female", 1: "Male"}),
            ui.input_select("AgeCategory", "Age Group", 
                            {0: 'Age 18 to 24', 1: 'Age 25 to 29', 2: 'Age 30 to 34', 
                             3: 'Age 35 to 39', 4: 'Age 40 to 44', 5: 'Age 45 to 49', 
                             6: 'Age 50 to 54', 7: 'Age 55 to 59', 8: 'Age 60 to 64', 
                             9: 'Age 65 to 69', 10: 'Age 70 to 74', 11: 'Age 75 to 79', 
                             12: 'Age 80 or older'}),
            ui.input_numeric("PhysicalHealthDays", "Physical Health Days", value=0),
            ui.input_numeric("MentalHealthDays", "Mental Health Days", value=0),
            ui.input_select("LastCheckupTime", "When was your checkup?", 
                            {0: "Within past year (anytime less than 12 months ago)", 
                             1: "Within past 2 years (1 year but less than 2 years ago)", 
                             2: "Within past 5 years (2 years but less than 5 years ago)", 
                             3: "5 or more years ago"}),
            ui.input_select("PhysicalActivities", "Physical Activities", {0: "No", 1: "Yes"}),
            ui.input_numeric("SleepHours", "Sleep Hours", value=0),
            ui.input_select("RemovedTeeth", "Have you had any teeth removed?", {0: "No", 1: "Yes"}),
            ui.input_select("GeneralHealth", "General Health", 
                            {0: "Poor", 1: "Fair", 2: "Good", 3: "Very good", 4: "Excellent"}),
            ui.input_numeric("HeightInMeters", "Height (in meters)", value=0),
            ui.input_numeric("WeightInKilograms", "Weight (in kilograms)", value=0),
            ui.input_select("SmokerStatus", "Smoking Status", 
                            {0: "Never smoked", 1: "Former smoker", 
                             2: "Current smoker - now smokes some days", 
                             3: "Current smoker - now smokes every day"}),
            ui.input_select("HIVTesting", "HIV Testing", {0: "No", 1: "Yes"}),
            ui.input_select("FluVaxLast12", "Flu Vax Last 12 Months", {0: "No", 1: "Yes"}),
            ui.input_select("PneumoVaxEver", "Pneumo Vax Ever", {0: "No", 1: "Yes"}),
            ui.input_select("TetanusLast10Tdap", "Tetanus Last 10 Years Tdap", 
                            {0: "No, did not receive any tetanus shot in the past 10 years", 
                             1: "Yes, received tetanus shot but not sure what type", 
                             2: "Yes, received Tdap in the past 10 years"}),
            ui.input_select("HighRiskLastYear", "High Risk Last Year", {0: "No", 1: "Yes"}),
            ui.input_select("CovidPos", "Covid Positive", {0: "No", 1: "Yes"}),
            ui.input_select("HadAsthma", "Had Asthma", {0: "No", 1: "Yes"}),
            ui.input_select("HadSkinCancer", "Had Skin Cancer", {0: "No", 1: "Yes"}),
            ui.input_select("HadCOPD", "Had COPD", {0: "No", 1: "Yes"}),
            ui.input_select("HadDepressiveDisorder", "Had Depressive Disorder", {0: "No", 1: "Yes"}),
            ui.input_select("HadKidneyDisease", "Had Kidney Disease", {0: "No", 1: "Yes"}),
            ui.input_select("HadArthritis", "Had Arthritis", {0: "No", 1: "Yes"}),
            ui.input_select("HadDiabetes", "Had Diabetes", {0: "No", 1: "Yes"}),
            ui.input_select("DeafOrHardOfHearing", "Deaf or Hard of Hearing", {0: "No", 1: "Yes"}),
            ui.input_select("BlindOrVisionDifficulty", "Blind or Vision Difficulty", {0: "No", 1: "Yes"}),
            ui.input_select("DifficultyConcentrating", "Difficulty Concentrating", {0: "No", 1: "Yes"}),
            ui.input_select("DifficultyWalking", "Difficulty Walking", {0: "No", 1: "Yes"}),
            ui.input_select("DifficultyDressingBathing", "Difficulty Dressing or Bathing", {0: "No", 1: "Yes"}),
            ui.input_select("DifficultyErrands", "Difficulty Running Errands", {0: "No", 1: "Yes"}),
            ui.input_select("ChestScan", "Have you ever had a chest scan?", {0: "No", 1: "Yes"}),
            ui.input_select("RaceEthnicityCategory", "Race/Ethnicity Category", 
                            {0: "White", 1: "Black", 2: "Asian", 3: "American Indian/Alaskan Native", 
                             4: "Hispanic", 5: "Other"}),
            ui.input_select("AlcoholDrinkers", "Alcohol Drinkers", 
                            {0: "No", 1: "Yes"})
        ),
        ui.panel_main(
            ui.markdown(
                """
                ### Heart Disease Risk Score (0 - 100)
                """
            ),
            ui.output_text_verbatim("txt", placeholder="Risk Score"),
            ui.output_image("risk_score_plot", width="60%", height="30%")
        )
    ),
)

# Server section
def server(input, output, session):
    @output
    @render.text
    def txt():
        # Collect input data
        input_data = pd.DataFrame([{
            'State': df['State'].unique()[input.State()],
            'Sex': df['Sex'].unique()[input.Sex()],
            'AgeCategory': df['AgeCategory'].unique()[input.AgeCategory()],
            'PhysicalHealthDays': input.PhysicalHealthDays(),
            'MentalHealthDays': input.MentalHealthDays(),
            'LastCheckupTime': df['LastCheckupTime'].unique()[input.LastCheckupTime()],
            'PhysicalActivities': df['PhysicalActivities'].unique()[input.PhysicalActivities()],
            'SleepHours': input.SleepHours(),
            'RemovedTeeth': df['RemovedTeeth'].unique()[input.RemovedTeeth()],
            'GeneralHealth': df['GeneralHealth'].unique()[input.GeneralHealth()],
            'HeightInMeters': input.HeightInMeters(),
            'WeightInKilograms': input.WeightInKilograms(),
            'SmokerStatus': df['SmokerStatus'].unique()[input.SmokerStatus()],
            'HIVTesting': df['HIVTesting'].unique()[input.HIVTesting()],
            'FluVaxLast12': df['FluVaxLast12'].unique()[input.FluVaxLast12()],
            'PneumoVaxEver': df['PneumoVaxEver'].unique()[input.PneumoVaxEver()],
            'TetanusLast10Tdap': df['TetanusLast10Tdap'].unique()[input.TetanusLast10Tdap()],
            'HighRiskLastYear': df['HighRiskLastYear'].unique()[input.HighRiskLastYear()],
            'CovidPos': df['CovidPos'].unique()[input.CovidPos()],
            'HadAsthma': df['HadAsthma'].unique()[input.HadAsthma()],
            'HadSkinCancer': df['HadSkinCancer'].unique()[input.HadSkinCancer()],
            'HadCOPD': df['HadCOPD'].unique()[input.HadCOPD()],
            'HadDepressiveDisorder': df['HadDepressiveDisorder'].unique()[input.HadDepressiveDisorder()],
            'HadKidneyDisease': df['HadKidneyDisease'].unique()[input.HadKidneyDisease()],
            'HadArthritis': df['HadArthritis'].unique()[input.HadArthritis()],
            'HadDiabetes': df['HadDiabetes'].unique()[input.HadDiabetes()],
            'DeafOrHardOfHearing': df['DeafOrHardOfHearing'].unique()[input.DeafOrHardOfHearing()],
            'BlindOrVisionDifficulty': df['BlindOrVisionDifficulty'].unique()[input.BlindOrVisionDifficulty()],
            'DifficultyConcentrating': df['DifficultyConcentrating'].unique()[input.DifficultyConcentrating()],
            'DifficultyWalking': df['DifficultyWalking'].unique()[input.DifficultyWalking()],
            'DifficultyDressingBathing': df['DifficultyDressingBathing'].unique()[input.DifficultyDressingBathing()],
            'DifficultyErrands': df['DifficultyErrands'].unique()[input.DifficultyErrands()],
            'ChestScan': df['ChestScan'].unique()[input.ChestScan()],
            'RaceEthnicityCategory': df['RaceEthnicityCategory'].unique()[input.RaceEthnicityCategory()],
            'AlcoholDrinkers': df['AlcoholDrinkers'].unique()[input.AlcoholDrinkers()],
            'BMI': input.WeightInKilograms() / (input.HeightInMeters() ** 2)
        }])

        # Make prediction
        prediction = pipeline.predict_proba(input_data)[0][1] * 100
        return f"Your risk score for CVD is {prediction:.2f}"

    @output
    @render.image
    def risk_score_plot():
        from pathlib import Path

        dir = Path(__file__).resolve().parent
        img = {"src": str(dir / "output.png"), "width": "100%", "height": "100%"}
        return img

app = App(app_ui, server)

if __name__ == '__main__':
    app.run()