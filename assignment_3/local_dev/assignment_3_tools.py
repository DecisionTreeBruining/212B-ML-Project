import os
import pickle
import pandas as pd
import polars as pl
from scipy.sparse import issparse
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

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

def parquet_extract(path, extract_list):
    """
    Lazy read specific or all parquet files in a folder.

    Args:
        path (str): Path to the folder containing parquet files.
        extract_list (list of str or None): List of file base names to extract
                                            (excluding `.parquet` extension).
                                            If None, all files are extracted.
    
    Returns:
        dict: A dictionary of lazy dataframes where keys are base file names.
    """
    pd_frames_dict = {}
    for filename in os.listdir(path):
        if filename.endswith(".parquet"):
            key = os.path.splitext(filename)[0]
            if extract_list is None or key in extract_list:
                file_path = os.path.join(path, filename)
                pd_frame = pl.scan_parquet(file_path).collect().to_pandas()
                pd_frames_dict[key] = pd_frame

    return pd_frames_dict

def dict_to_parquet(lazydict, drive_path):
    """
    Lazy read all parquet files in a folder.
    ---
    Args: 
        lazydict: A dictionary of lazyframes
    Return: 
        None
    """
    for name, df in lazydict.items():
        df.collect().write_parquet(f"{drive_path}{name}.parquet")

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
    """
    Encodes the dataset using one-hot encoding for general categorical columns and 
    ordinal encoding for specified columns with predefined categories. Splits the data into training and test datasets.

    Parameters:
        df (pandas.DataFrame): DataFrame to encode and split.

    Returns:
      X_train_encoded (DataFrame): Encoded training features.
      X_test_encoded (DataFrame): Encoded test features.
      y_train (Series): Training target variable.
      y_test (Series): Test target variable.

    """
    # convert health days from float to int
    # df['PhysicalHealthDays'] = df['PhysicalHealthDays'].astype(int)
    # df['MentalHealthDays'] = df['MentalHealthDays'].astype(int)
    
    # Define the features and target
    X = df.drop('HadHeartDisease', axis=1)
    y = df['HadHeartDisease']

    # data split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                        test_size = 0.25, 
                                        random_state = 69,
                                        stratify = y)

    # Define which columns to one-hot encode and which to label encode
    categorical_cols = X.select_dtypes(include=['object']).columns
    one_hot_cols = categorical_cols.drop(['GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 
                                          'AgeCategory', 'SmokerStatus', 'ECigaretteUsage'])
    
    # define the columns with specific encoding
    comp_labels = {
        "GeneralHealth": ['Poor', 'Fair', 'Good', 
                          'Very good', 'Excellent'],

        "LastCheckupTime": ['5 or more years ago',
                            'Within past 5 years (2 years but less than 5 years ago)',
                            'Within past 2 years (1 year but less than 2 years ago)',
                            'Within past year (anytime less than 12 months ago)'],

        "RemovedTeeth": ['None of them', '1 to 5',
                         '6 or more, but not all', 'All'],

        "SmokerStatus": ['Never smoked', 'Former smoker',
                         'Current smoker - now smokes some days',
                         'Current smoker - now smokes every day'],
                         
        "ECigaretteUsage": ['Never used e-cigarettes in my entire life',
                            'Not at all (right now)',
                            'Use them some days',
                            'Use them every day']
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

    # fit another label encoder on y
    y_label_encoder = OrdinalEncoder()
    y_train_encoded = y_label_encoder.fit_transform(pd.DataFrame(y_train))
    y_test_encoded = y_label_encoder.transform(pd.DataFrame(y_test))

    # Handle sparse matrix if necessary
    if issparse(X_train_encoded):
        X_train_encoded = X_train_encoded.toarray()
    if issparse(X_test_encoded):
        X_test_encoded = X_test_encoded.toarray()

    # Convert the sparse matrix to DataFrame and specify column names
    X_columns = preprocessor.get_feature_names_out()
    y_column = y_label_encoder.get_feature_names_out()
    X_train_encoded = pd.DataFrame(X_train_encoded, columns=X_columns, index=X_train.index)
    X_test_encoded = pd.DataFrame(X_test_encoded, columns=X_columns, index=X_test.index)
    y_train_encoded = pd.DataFrame(y_train_encoded, columns=y_column, index=y_train.index)
    y_test_encoded = pd.DataFrame(y_test_encoded, columns=y_column, index=y_test.index)
    
    return X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded

def rand_under_sample(X_train, y_train, n_to_y = 1):
    """
    Balance by Random undersampling
    ---
    Args:
        X_train (pd.dataframe): The training set of predictors
        y_train (pd.dataframe): The training set of target feature values
        n_to_y (float): The ratio of no to yes
    Return:
        X_train_res (pd.dataframe): The balanced training set of predictors
        y_train_res (pd.dataframe): The balanced training set of target feature values
    """
    yes_count = y_train.value_counts()[1]
    no_count = yes_count * n_to_y
    sampling_strategy = {0: no_count, 1: yes_count}
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy) 
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

def rand_over_sample(X_train, y_train, y_to_n = 1):
    """
    Balance by Random undersampling
    ---
    Args:
        X_train (pd.dataframe): The training set of predictors
        y_train (pd.dataframe): The training set of target feature values
        y_to_n (float): The ratio of no to yes
    Return:
        X_train_res (pd.dataframe): The balanced training set of predictors
        y_train_res (pd.dataframe): The balanced training set of target feature values
    """
    yes_count = y_train.value_counts()[0]
    no_count = yes_count * y_to_n
    sampling_strategy = {0: no_count, 1: yes_count}
    rus = RandomOverSampler(sampling_strategy=sampling_strategy) 
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
    return X_train_res, y_train_res

def balanced_dict(X_train, y_train, ratios):
    balanced_X_train = dict()
    balanced_y_train = dict()
    for i in ratios:
        under_name = f"Under_Sample_{i}:1"
        over_name = f"Over_Sample_1:{i}"
        balanced_X_train[under_name], balanced_y_train[under_name] = rand_under_sample(X_train, y_train, i)
        balanced_X_train[over_name], balanced_y_train[over_name] = rand_over_sample(X_train, y_train, i)
    return balanced_X_train, balanced_y_train