from shiny import App, render, ui, reactive
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from app_tools import pickle_to_dict
import pickle

# Load the model
data_path = '../models/'
pickle_dict = pickle_to_dict(data_path)

pipeline = Pipeline(steps=[
    ('preprocessor', pickle_dict['preprocessor']),
    ('scaler', pickle_dict['standard_scaler']),
    ('model', pickle_dict['best_model'])
])

## List of options
state_list = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 
    'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia', 
    'Guam', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 
    'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 
    'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 
    'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 
    'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 
    'Oregon', 'Pennsylvania', 'Puerto Rico', 'Rhode Island', 'South Carolina', 
    'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virgin Islands', 
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming'
]

sex_list = ["Female", "Male"]

age_list = [
    'Age 18 to 24', 'Age 25 to 29', 'Age 30 to 34', 
    'Age 35 to 39', 'Age 40 to 44', 'Age 45 to 49', 
    'Age 50 to 54', 'Age 55 to 59', 'Age 60 to 64', 
    'Age 65 to 69', 'Age 70 to 74', 'Age 75 to 79', 
    'Age 80 or older'
]

lastcheckup_list = [
    "Within past year (anytime less than 12 months ago)", 
    "Within past 2 years (1 year but less than 2 years ago)", 
    "Within past 5 years (2 years but less than 5 years ago)", 
    "5 or more years ago"
]

yes_no_list = ["No", "Yes"]

gen_health_list = ["Poor", "Fair", "Good", "Very good", "Excellent"]

smoker_list = [
    "Never smoked", "Former smoker", 
    "Current smoker - now smokes some days", 
    "Current smoker - now smokes every day"
]

tdap_list = [
    "No, did not receive any tetanus shot in the past 10 years", 
    "Yes, received tetanus shot but not sure what type", 
    "Yes, received Tdap in the past 10 years"
]

race_list = [
    "White", "Black", "Asian", 
    "American Indian/Alaskan Native", 
    "Hispanic", "Other"
]

# UI section starts from here 
app_ui = ui.page_fluid(
    ui.markdown(
        """
        ## Heart Disease Prediction Model
        """
    ),
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_action_button("submit", "Submit"),
            ui.input_select("State", "Which state do you live in?", state_list),
            ui.input_select("Sex", "Sex", sex_list),
            ui.input_select("AgeCategory", "Age Group", age_list),
            ui.input_numeric("PhysicalHealthDays", "Physical Health Days", value=0),
            ui.input_numeric("MentalHealthDays", "Mental Health Days", value=0),
            ui.input_select("LastCheckupTime", "When was your checkup?", lastcheckup_list),
            ui.input_select("PhysicalActivities", "Physical Activities", yes_no_list),
            ui.input_numeric("SleepHours", "Sleep Hours", value=0),
            ui.input_select("RemovedTeeth", "Have you had any teeth removed?", yes_no_list),
            ui.input_select("GeneralHealth", "General Health", gen_health_list),
            ui.input_numeric("HeightInMeters", "Height (in meters)", value=0),
            ui.input_numeric("WeightInKilograms", "Weight (in kilograms)", value=0),
            ui.input_select("SmokerStatus", "Smoking Status", smoker_list),
            ui.input_select("HIVTesting", "HIV Testing",yes_no_list),
            ui.input_select("FluVaxLast12", "Flu Vax Last 12 Months", yes_no_list),
            ui.input_select("PneumoVaxEver", "Pneumo Vax Ever", yes_no_list),
            ui.input_select("TetanusLast10Tdap", "Tetanus Last 10 Years Tdap", tdap_list),
            ui.input_select("HighRiskLastYear", "High Risk Last Year", yes_no_list),
            ui.input_select("CovidPos", "Covid Positive", yes_no_list),
            ui.input_select("HadAsthma", "Had Asthma", yes_no_list),
            ui.input_select("HadSkinCancer", "Had Skin Cancer", yes_no_list),
            ui.input_select("HadCOPD", "Had COPD", yes_no_list),
            ui.input_select("HadDepressiveDisorder", "Had Depressive Disorder", yes_no_list),
            ui.input_select("HadKidneyDisease", "Had Kidney Disease", yes_no_list),
            ui.input_select("HadArthritis", "Had Arthritis", yes_no_list),
            ui.input_select("HadDiabetes", "Had Diabetes", yes_no_list),
            ui.input_select("DeafOrHardOfHearing", "Deaf or Hard of Hearing", yes_no_list),
            ui.input_select("BlindOrVisionDifficulty", "Blind or Vision Difficulty", yes_no_list),
            ui.input_select("DifficultyConcentrating", "Difficulty Concentrating", yes_no_list),
            ui.input_select("DifficultyWalking", "Difficulty Walking", yes_no_list),
            ui.input_select("DifficultyDressingBathing", "Difficulty Dressing or Bathing", yes_no_list),
            ui.input_select("DifficultyErrands", "Difficulty Running Errands", yes_no_list),
            ui.input_select("ChestScan", "Have you ever had a chest scan?", yes_no_list),
            ui.input_select("RaceEthnicityCategory", "Race/Ethnicity Category", race_list),
            ui.input_select("AlcoholDrinkers", "Alcohol Drinkers", yes_no_list)
        ),
        ui.panel_main(
            ui.markdown(
                """
                ### Heart Disease Risk Score (0 - 100)
                """
            ),
            ui.output_text_verbatim("txt", placeholder="Risk Score"),
            ui.output_image("risk_score_plot", width="60%", height="30%") #output_plot
        )
    ),
)

# Server section
def server(input, output, session):
    prediction_value = reactive.Value("Submit your information to get the risk score.")
    @reactive.Effect
    @reactive.event(input.submit)
    def submit_info():
        # Convert input values to integers
        state = int(input.State())
        sex = int(input.Sex())
        age_category = int(input.AgeCategory())
        last_checkup_time = int(input.LastCheckupTime())
        physical_activities = int(input.PhysicalActivities())
        removed_teeth = int(input.RemovedTeeth())
        general_health = int(input.GeneralHealth())
        smoker_status = int(input.SmokerStatus())
        hiv_testing = int(input.HIVTesting())
        flu_vax_last12 = int(input.FluVaxLast12())
        pneumo_vax_ever = int(input.PneumoVaxEver())
        tetanus_last10_tdap = int(input.TetanusLast10Tdap())
        high_risk_last_year = int(input.HighRiskLastYear())
        covid_pos = int(input.CovidPos())
        had_asthma = int(input.HadAsthma())
        had_skin_cancer = int(input.HadSkinCancer())
        had_copd = int(input.HadCOPD())
        had_depressive_disorder = int(input.HadDepressiveDisorder())
        had_kidney_disease = int(input.HadKidneyDisease())
        had_arthritis = int(input.HadArthritis())
        had_diabetes = int(input.HadDiabetes())
        deaf_or_hard_of_hearing = int(input.DeafOrHardOfHearing())
        blind_or_vision_difficulty = int(input.BlindOrVisionDifficulty())
        difficulty_concentrating = int(input.DifficultyConcentrating())
        difficulty_walking = int(input.DifficultyWalking())
        difficulty_dressing_bathing = int(input.DifficultyDressingBathing())
        difficulty_errands = int(input.DifficultyErrands())
        chest_scan = int(input.ChestScan())
        race_ethnicity_category = int(input.RaceEthnicityCategory())
        alcohol_drinkers = int(input.AlcoholDrinkers())

        # Collect input data
        input_data = pd.DataFrame([{
            'State': state_mapping[state],
            'Sex': sex_mapping[sex],
            'AgeCategory': age_category_mapping[age_category],
            'PhysicalHealthDays': input.PhysicalHealthDays(),
            'MentalHealthDays': input.MentalHealthDays(),
            'LastCheckupTime': last_checkup_time_mapping[last_checkup_time],
            'PhysicalActivities': physical_activities_mapping[physical_activities],
            'SleepHours': input.SleepHours(),
            'RemovedTeeth': removed_teeth_mapping[removed_teeth],
            'GeneralHealth': general_health_mapping[general_health],
            'HeightInMeters': input.HeightInMeters(),
            'WeightInKilograms': input.WeightInKilograms(),
            'SmokerStatus': smoker_status_mapping[smoker_status],
            'HIVTesting': yes_no_mapping[hiv_testing],
            'FluVaxLast12': yes_no_mapping[flu_vax_last12],
            'PneumoVaxEver': yes_no_mapping[pneumo_vax_ever],
            'TetanusLast10Tdap': tetanus_last10_tdap,
            'HighRiskLastYear': yes_no_mapping[high_risk_last_year],
            'CovidPos': yes_no_mapping[covid_pos],
            'HadAsthma': yes_no_mapping[had_asthma],
            'HadSkinCancer': yes_no_mapping[had_skin_cancer],
            'HadCOPD': yes_no_mapping[had_copd],
            'HadDepressiveDisorder': yes_no_mapping[had_depressive_disorder],
            'HadKidneyDisease': yes_no_mapping[had_kidney_disease],
            'HadArthritis': yes_no_mapping[had_arthritis],
            'HadDiabetes': yes_no_mapping[had_diabetes],
            'DeafOrHardOfHearing': yes_no_mapping[deaf_or_hard_of_hearing],
            'BlindOrVisionDifficulty': yes_no_mapping[blind_or_vision_difficulty],
            'DifficultyConcentrating': yes_no_mapping[difficulty_concentrating],
            'DifficultyWalking': yes_no_mapping[difficulty_walking],
            'DifficultyDressingBathing': yes_no_mapping[difficulty_dressing_bathing],
            'DifficultyErrands': yes_no_mapping[difficulty_errands],
            'ChestScan': yes_no_mapping[chest_scan],
            'RaceEthnicityCategory': race_ethnicity_mapping[race_ethnicity_category],
            'AlcoholDrinkers': yes_no_mapping[alcohol_drinkers],
            'BMI': input.WeightInKilograms() / (input.HeightInMeters() ** 2)
        }])
        # Make prediction
        prediction = pipeline.predict_proba(input_data)[0][1] * 100
        prediction_value.set(f"Your risk score for CVD is {prediction:.2f}")

    @output
    @render.text
    def txt():
        return prediction_value.get()

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