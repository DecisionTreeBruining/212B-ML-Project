import os  # Import os for interacting with the operating system
import pickle  # Import pickle for serializing and deserializing Python objects
import pandas as pd  # Import pandas for data manipulation and analysis
import polars as pl  # Import polars for handling data in a fast and memory-efficient way
from sklearn.compose import ColumnTransformer  # Import ColumnTransformer for preprocessing
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # Import encoders and scaler
from sklearn.model_selection import train_test_split  # Import function to split data into training and test sets
from scipy.sparse import issparse  # Import issparse to check if a matrix is sparse

def parquet_to_dict(path):
    """
    Lazy read all parquet files in a folder.
    ---
    Args: 
        Path: Relative Path to folder
    Return: 
        lazy_frames_dict: Dictionary of lazy dataframes
    """
    lazy_frames_dict = {}
    for filename in os.listdir(path): # Iterate over each file
        if filename.endswith(".parquet"):
            file_path = os.path.join(path, filename) # File Path
            lazy_frame = pl.scan_parquet(file_path) # Lazy read
            key = os.path.splitext(filename)[0] # key = filename
            lazy_frames_dict[key] = lazy_frame # Add lazyframe to dictionary
    return lazy_frames_dict

def pickle_to_dict(path):
    """
    Read all pickle files in a folder and return a dictionary of objects.
    
    Args:
        path (str): Relative or absolute path to the folder containing pickle files.
        
    Returns:
        dict: A dictionary where each key is the filename (without extension) and the value is the loaded object.
    """
    pickle_rick = {}
    for filename in os.listdir(path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(path, filename)
            with open(file_path, 'rb') as file:
                pickle_rick[os.path.splitext(filename)[0]] = pickle.load(file)
    return pickle_rick

def encode_cvd_var(df):
    # Define the features and target
    X = df.drop('HadHeartDisease', axis=1)
    y = df['HadHeartDisease']
    y = df['HadHeartDisease'].map({'Yes': 1, 'No': 0})
  
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.25, 
        random_state=69,
        stratify=y
    )

    # Define which columns to one-hot encode and which to label encode
    categorical_cols = X.select_dtypes(include=['object']).columns
    one_hot_cols = categorical_cols.drop([
        'GeneralHealth', 
        'LastCheckupTime',
        'RemovedTeeth', 
        'AgeCategory', 
        'SmokerStatus', 
        'ECigaretteUsage'
    ])
    
    # Define the columns with specific encoding
    comp_labels = {
        "GeneralHealth": [
            'Poor', 
            'Fair', 
            'Good', 
            'Very good', 
            'Excellent'
        ],
        "LastCheckupTime": [
            '5 or more years ago',
            'Within past 5 years (2 years but less than 5 years ago)',
            'Within past 2 years (1 year but less than 2 years ago)',
            'Within past year (anytime less than 12 months ago)'
        ],
        "RemovedTeeth": [
            'None of them', 
            '1 to 5',
            '6 or more, but not all', 
            'All'
        ],
        "SmokerStatus": [
            'Never smoked', 
            'Former smoker',
            'Current smoker - now smokes some days',
            'Current smoker - now smokes every day'
        ],  
        "ECigaretteUsage": [
            'Never used e-cigarettes in my entire life',
            'Not at all (right now)',
            'Use them some days',
            'Use them every day'
        ]
    }

    label_encoders = [(key + '_label', OrdinalEncoder(categories=[value]), [key]) 
                      for key, value in comp_labels.items()]
    
    # Define the preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), one_hot_cols),
            ('label', OrdinalEncoder(), ['AgeCategory']),
        ] + label_encoders
        , remainder='passthrough'
    )
    
    # Fit the preprocessor on the training data only and transform both
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Save the preprocessor
    with open('../models/preprocessor.pkl', 'wb') as file:
        pickle.dump(preprocessor, file)

    # Handle sparse matrix if necessary
    if issparse(X_train_encoded):
        X_train_encoded = X_train_encoded.toarray()
    if issparse(X_test_encoded):
        X_test_encoded = X_test_encoded.toarray()

    # Fit and save the StandardScaler on the training data
    scaler = StandardScaler()
    X_train_encoded = scaler.fit_transform(X_train_encoded)
    X_test_encoded = scaler.transform(X_test_encoded)

    with open('../models/standard_scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    # Convert the array back to DataFrame and specify column names
    X_columns = preprocessor.get_feature_names_out()
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=X_columns, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=X_columns, index=X_test.index)
    y_train_encoded = pd.DataFrame(y_train)
    y_test_encoded = pd.DataFrame(y_test)
    
    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded

def load_and_transform_new_data(new_data):
    """
    Load the saved preprocessor and scaler, and transform new data.

    Parameters:
        new_data (pandas.DataFrame): DataFrame containing new data to be transformed.

    Returns:
        transformed_data (DataFrame): Transformed new data.
    """
    # Load the preprocessor
    with open('../models/preprocessor.pkl', 'rb') as file:
        preprocessor = pickle.load(file)

    # Normalize the new data (as an example, assuming same preprocessing as before)
    new_data['GeneralHealth'] = new_data['GeneralHealth'].str.strip().str.lower()
    new_data['LastCheckupTime'] = new_data['LastCheckupTime'].str.strip().str.lower()
    new_data['RemovedTeeth'] = new_data['RemovedTeeth'].str.strip().str.lower()
    new_data['SmokerStatus'] = new_data['SmokerStatus'].str.strip().str.lower()
    new_data['ECigaretteUsage'] = new_data['ECigaretteUsage'].str.strip().str.lower()
    new_data['AgeCategory'] = new_data['AgeCategory'].str.strip().str.lower()

    # Transform the new data
    new_data_encoded = preprocessor.transform(new_data)

    # Load the scaler
    with open('../models/standard_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    # Handle sparse matrix if necessary
    if issparse(new_data_encoded):
        new_data_encoded = new_data_encoded.toarray()

    # Standardize the new data
    new_data_encoded = scaler.transform(new_data_encoded)

    # Convert the array back to DataFrame and specify column names
    X_columns = preprocessor.get_feature_names_out()
    transformed_data = pd.DataFrame(new_data_encoded, columns=X_columns, index=new_data.index)

    return transformed_data