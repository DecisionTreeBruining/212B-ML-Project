from shiny import App, render, ui, reactive
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from shinyswatch import theme
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
    'White only, Non-Hispanic', 'Black only, Non-Hispanic',
    'Other race only, Non-Hispanic', 'Multiracial, Non-Hispanic',
    'Hispanic']

ECigarette_list = [
    'Never used e-cigarettes in my entire life',
    'Not at all (right now)',
    'Use them some days',
    'Use them every day'
]

teeth_list = [
    'None of them', 
    '1 to 5',
    '6 or more, but not all', 
    'All'
]

diabetes_list = ['Yes', 'No', 'No, pre-diabetes or borderline diabetes',
       'Yes, but only during pregnancy (female)']

# UI section starts from here 
app_ui = ui.page_fluid(
    theme.flatly(),  # Applying the flatly theme for a blue look
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
            ui.input_select("ECigaretteUsage", "What's your E-Cigarette Usage?", ECigarette_list),
            ui.input_select("PhysicalActivities", "Physical Activities", yes_no_list), #checked
            ui.input_numeric("SleepHours", "Sleep Hours", value=0),
            ui.input_select("RemovedTeeth", "Have you had any teeth removed?", teeth_list),
            ui.input_select("GeneralHealth", "General Health", gen_health_list),
            ui.input_numeric("HeightInMeters", "Height (in meters)", value=0),
            ui.input_numeric("WeightInKilograms", "Weight (in kilograms)", value=0),
            ui.input_select("SmokerStatus", "Smoking Status", smoker_list),
            ui.input_select("HIVTesting", "HIV Testing",yes_no_list), #checked
            ui.input_select("FluVaxLast12", "Flu Vax Last 12 Months", yes_no_list), #checked
            ui.input_select("PneumoVaxEver", "Pneumo Vax Ever", yes_no_list), #checked
            ui.input_select("TetanusLast10Tdap", "Tetanus Last 10 Years Tdap", tdap_list), #checked
            ui.input_select("HighRiskLastYear", "High Risk Last Year", yes_no_list), #checked
            ui.input_select("CovidPos", "Covid Positive", yes_no_list), #checked
            ui.input_select("HadAsthma", "Had Asthma", yes_no_list), #checked
            ui.input_select("HadSkinCancer", "Had Skin Cancer", yes_no_list), #checked
            ui.input_select("HadCOPD", "Had COPD", yes_no_list), #checked
            ui.input_select("HadDepressiveDisorder", "Had Depressive Disorder", yes_no_list), #checked
            ui.input_select("HadKidneyDisease", "Had Kidney Disease", yes_no_list), #checked
            ui.input_select("HadArthritis", "Had Arthritis", yes_no_list), #checked
            ui.input_select("HadDiabetes", "Had Diabetes", diabetes_list), #checked
            ui.input_select("DeafOrHardOfHearing", "Deaf or Hard of Hearing", yes_no_list), #checked
            ui.input_select("BlindOrVisionDifficulty", "Blind or Vision Difficulty", yes_no_list), #checked
            ui.input_select("DifficultyConcentrating", "Difficulty Concentrating", yes_no_list), #checked
            ui.input_select("DifficultyWalking", "Difficulty Walking", yes_no_list), #checked
            ui.input_select("DifficultyDressingBathing", "Difficulty Dressing or Bathing", yes_no_list), #checked
            ui.input_select("DifficultyErrands", "Difficulty Running Errands", yes_no_list), #checked
            ui.input_select("ChestScan", "Have you ever had a chest scan?", yes_no_list), #checked
            ui.input_select("RaceEthnicityCategory", "Race/Ethnicity Category", race_list), #checked
            ui.input_select("AlcoholDrinkers", "Alcohol Drinkers", yes_no_list) #checked
        ),
        ui.panel_main(
            ui.markdown(
                """
                ### Heart Disease Risk Score (0 - 100)
                """
            ),
            ui.output_text_verbatim("txt", placeholder="Risk Score"),
            ui.div(
                ui.output_plot("risk_score_plot"), 
                style="height: 500px; width: 100%;"
            ) #output_plot
        )
    ),
)

# Server section
def server(input, output, session):
    prediction_value = reactive.Value("Submit your information to get the risk score.")
    prediction = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.submit)
    def submit_info():
        # Collect input data
        input_data = pd.DataFrame([{
            'State': input.State(),
            'Sex': input.Sex(),
            'AgeCategory': input.AgeCategory(),
            'PhysicalHealthDays': input.PhysicalHealthDays(),
            'LastCheckupTime': input.LastCheckupTime(),
            'ECigaretteUsage': input.ECigaretteUsage(),
            'PhysicalActivities': input.PhysicalActivities(),
            'MentalHealthDays': input.MentalHealthDays(),
            'SleepHours': input.SleepHours(),
            'RemovedTeeth': input.RemovedTeeth(),
            'GeneralHealth': input.GeneralHealth(),
            'HeightInMeters': input.HeightInMeters(),
            'WeightInKilograms': input.WeightInKilograms(),
            'SmokerStatus': input.SmokerStatus(),
            'HIVTesting': input.HIVTesting(),
            'FluVaxLast12': input.FluVaxLast12(),
            'PneumoVaxEver': input.PneumoVaxEver(),
            'TetanusLast10Tdap': input.TetanusLast10Tdap(),
            'HighRiskLastYear': input.HighRiskLastYear(),
            'CovidPos': input.CovidPos(),
            'HadAsthma': input.HadAsthma(),
            'HadSkinCancer': input.HadSkinCancer(),
            'HadCOPD': input.HadCOPD(),
            'HadDepressiveDisorder': input.HadDepressiveDisorder(),
            'HadKidneyDisease': input.HadKidneyDisease(),
            'HadArthritis': input.HadArthritis(),
            'HadDiabetes': input.HadDiabetes(),
            'DeafOrHardOfHearing': input.DeafOrHardOfHearing(),
            'BlindOrVisionDifficulty': input.BlindOrVisionDifficulty(),
            'DifficultyConcentrating': input.DifficultyConcentrating(),
            'DifficultyWalking': input.DifficultyWalking(),
            'DifficultyDressingBathing': input.DifficultyDressingBathing(),
            'DifficultyErrands': input.DifficultyErrands(),
            'ChestScan': input.ChestScan(),
            'RaceEthnicityCategory': input.RaceEthnicityCategory(),
            'AlcoholDrinkers': input.AlcoholDrinkers()
        }])
        # Make prediction
        pred = pipeline.predict_proba(input_data)[0][1] * 100
        prediction_value.set(f"Your risk score for CVD is {pred:.2f}")
        prediction.set(pred)

    @output
    @render.text
    def txt():
        return prediction_value.get()
    
    @output
    @render.plot
    def risk_score_plot():
        data = pickle_dict['pop_pred_data']
        user = prediction.get()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data, kde=True, stat='density', ax=ax)
        ax.set_title('Distribution of Risk Scores')
        ax.set_xlabel('Risk Score')
        ax.set_ylabel('Prevalence')
        ax = fig.gca()
        if user is not None:
            ax.axvline(user, color='red', linestyle='-', linewidth=2)
            ax.text(user + 1, 0.02, 'Your Risk of Heart Disease', color='red', rotation=90)
        return fig

        # fig, ax = plt.subplots(figsize=(10, 6))
        # fig = pickle_dict['pop_plot']
        # ax = fig.gca()
        # pred = prediction.get()
        # if pred is not None:
        #     ax.axvline(pred, color='red', linestyle='-', linewidth=2)
        #     ax.text(pred + 1, 0.02, 'You are here', color='red', rotation=90)
        # return fig

app = App(app_ui, server)

if __name__ == '__main__':
    app.run()
