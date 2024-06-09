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

with open('../models/scaler.pkl', 'rb') as file:
    standard_scaler = pickle.load(file)

# Define the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', standard_scaler),
    ('model', best_model)
])
  
# Unique classes for each variable
state_classes = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
                 'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia',
                 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
                 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
                 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
                 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
                 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
                 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia',
                 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming', 'Guam', 'Puerto Rico',
                 'Virgin Islands']

sex_classes = ['Female', 'Male']
general_health_classes = ['Poor', 'Fair', 'Good', 
                          'Very good', 'Excellent']
last_checkup_time_classes = ['5 or more years ago',
                             'Within past 5 years (2 years but less than 5 years ago)',
                             'Within past 2 years (1 year but less than 2 years ago)',
                             'Within past year (anytime less than 12 months ago)']
physical_activities_classes = ['No', 'Yes']
removed_teeth_classes = ['None of them', '1 to 5',
                         '6 or more, but not all', 'All']
yes_no_classes = ['No', 'Yes']
had_diabetes_classes = ['Yes', 'No', 'No, pre-diabetes or borderline diabetes', 'Yes, but only during pregnancy (female)']
smoker_status_classes = ['Never smoked', 'Former smoker',
                         'Current smoker - now smokes some days',
                         'Current smoker - now smokes every day']
e_cigarette_usage_classes = ['Never used e-cigarettes in my entire life',
                             'Not at all (right now)',
                             'Use them some days',
                             'Use them every day']
race_ethnicity_category_classes = ['White only, Non-Hispanic', 'Black only, Non-Hispanic',
                                   'Other race only, Non-Hispanic', 'Multiracial, Non-Hispanic', 'Hispanic']
age_category_classes = ['Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 'Age 35 to 39', 'Age 40 to 44', 
                        'Age 45 to 49', 'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 'Age 65 to 69', 
                        'Age 70 to 74', 'Age 75 to 79', 'Age 80 or older']
tetanus_last_10_classes = ['Yes, received tetanus shot but not sure what type',
                           'No, did not receive any tetanus shot in the past 10 years',
                           'Yes, received Tdap', 'Yes, received tetanus shot, but not Tdap']
covid_pos_classes = ['No', 'Yes', 'Tested positive using home test without a health professional']

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
                            {i: state for i, state in enumerate(state_classes)}),
            ui.input_select("Sex", "Sex", {i: sex for i, sex in enumerate(sex_classes)}),
            ui.input_select("GeneralHealth", "General Health", 
                            {i: health for i, health in enumerate(general_health_classes)}),
            ui.input_numeric("PhysicalHealthDays", "Physical Health Days", value=0),
            ui.input_numeric("MentalHealthDays", "Mental Health Days", value=0),
            ui.input_select("LastCheckupTime", "When was your checkup?", 
                            {i: time for i, time in enumerate(last_checkup_time_classes)}),
            ui.input_select("PhysicalActivities", "Physical Activities", {i: activity for i, activity in enumerate(physical_activities_classes)}),
            ui.input_numeric("SleepHours", "Sleep Hours", value=0),
            ui.input_select("RemovedTeeth", "Have you had any teeth removed?", {i: teeth for i, teeth in enumerate(removed_teeth_classes)}),
            ui.input_select("HadAsthma", "Had Asthma", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("HadSkinCancer", "Had Skin Cancer", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("HadCOPD", "Had COPD", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("HadDepressiveDisorder", "Had Depressive Disorder", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("HadKidneyDisease", "Had Kidney Disease", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("HadArthritis", "Had Arthritis", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("HadDiabetes", "Had Diabetes", {i: diabetes for i, diabetes in enumerate(had_diabetes_classes)}),
            ui.input_select("DeafOrHardOfHearing", "Deaf or Hard of Hearing", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("BlindOrVisionDifficulty", "Blind or Vision Difficulty", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("DifficultyConcentrating", "Difficulty Concentrating", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("DifficultyWalking", "Difficulty Walking", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("DifficultyDressingBathing", "Difficulty Dressing or Bathing", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("DifficultyErrands", "Difficulty Running Errands", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("SmokerStatus", "Smoking Status", {i: status for i, status in enumerate(smoker_status_classes)}),
            ui.input_select("ECigaretteUsage", "E-Cigarette Usage", {i: usage for i, usage in enumerate(e_cigarette_usage_classes)}),
            ui.input_select("ChestScan", "Have you ever had a chest scan?", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("RaceEthnicityCategory", "Race/Ethnicity Category", {i: category for i, category in enumerate(race_ethnicity_category_classes)}),
            ui.input_select("AgeCategory", "Age Group", {i: age for i, age in enumerate(age_category_classes)}),
            ui.input_numeric("HeightInMeters", "Height (in meters)", value=0),
            ui.input_numeric("WeightInKilograms", "Weight (in kilograms)", value=0),
            ui.input_select("AlcoholDrinkers", "Alcohol Drinkers", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("HIVTesting", "HIV Testing", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("FluVaxLast12", "Flu Vax Last 12 Months", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("PneumoVaxEver", "Pneumo Vax Ever", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("TetanusLast10Tdap", "Tetanus Last 10 Years Tdap", {i: status for i, status in enumerate(tetanus_last_10_classes)}),
            ui.input_select("HighRiskLastYear", "High Risk Last Year", {i: answer for i, answer in enumerate(yes_no_classes)}),
            ui.input_select("CovidPos", "Covid Positive", {i: status for i, status in enumerate(covid_pos_classes)}),
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
            'State': state_classes[int(input.State())],
            'Sex': sex_classes[int(input.Sex())],
            'GeneralHealth': general_health_classes[int(input.GeneralHealth())],
            'PhysicalHealthDays': input.PhysicalHealthDays(),
            'MentalHealthDays': input.MentalHealthDays(),
            'LastCheckupTime': last_checkup_time_classes[int(input.LastCheckupTime())],
            'PhysicalActivities': physical_activities_classes[int(input.PhysicalActivities())],
            'SleepHours': input.SleepHours(),
            'RemovedTeeth': removed_teeth_classes[int(input.RemovedTeeth())],
            'HadAsthma': yes_no_classes[int(input.HadAsthma())],
            'HadSkinCancer': yes_no_classes[int(input.HadSkinCancer())],
            'HadCOPD': yes_no_classes[int(input.HadCOPD())],
            'HadDepressiveDisorder': yes_no_classes[int(input.HadDepressiveDisorder())],
            'HadKidneyDisease': yes_no_classes[int(input.HadKidneyDisease())],
            'HadArthritis': yes_no_classes[int(input.HadArthritis())],
            'HadDiabetes': had_diabetes_classes[int(input.HadDiabetes())],
            'DeafOrHardOfHearing': yes_no_classes[int(input.DeafOrHardOfHearing())],
            'BlindOrVisionDifficulty': yes_no_classes[int(input.BlindOrVisionDifficulty())],
            'DifficultyConcentrating': yes_no_classes[int(input.DifficultyConcentrating())],
            'DifficultyWalking': yes_no_classes[int(input.DifficultyWalking())],
            'DifficultyDressingBathing': yes_no_classes[int(input.DifficultyDressingBathing())],
            'DifficultyErrands': yes_no_classes[int(input.DifficultyErrands())],
            'SmokerStatus': smoker_status_classes[int(input.SmokerStatus())],
            'ECigaretteUsage': e_cigarette_usage_classes[int(input.ECigaretteUsage())],
            'ChestScan': yes_no_classes[int(input.ChestScan())],
            'RaceEthnicityCategory': race_ethnicity_category_classes[int(input.RaceEthnicityCategory())],
            'AgeCategory': age_category_classes[int(input.AgeCategory())],
            'HeightInMeters': input.HeightInMeters(),
            'WeightInKilograms': input.WeightInKilograms(),
            'AlcoholDrinkers': yes_no_classes[int(input.AlcoholDrinkers())],
            'HIVTesting': yes_no_classes[int(input.HIVTesting())],
            'FluVaxLast12': yes_no_classes[int(input.FluVaxLast12())],
            'PneumoVaxEver': yes_no_classes[int(input.PneumoVaxEver())],
            'TetanusLast10Tdap': tetanus_last_10_classes[int(input.TetanusLast10Tdap())],
            'HighRiskLastYear': yes_no_classes[int(input.HighRiskLastYear())],
            'CovidPos': covid_pos_classes[int(input.CovidPos())]
        }])


        # Apply preprocessing and scaling
        processed_data = pipeline.named_steps['preprocessor'].transform(input_data)
        scaled_data = pipeline.named_steps['scaler'].transform(processed_data)

        # Make prediction
        prediction = pipeline.named_steps['model'].predict_proba(scaled_data)[0][1] * 100
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