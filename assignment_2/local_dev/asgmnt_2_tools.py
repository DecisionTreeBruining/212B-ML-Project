import polars as pl # Lazy Dataframe Manipulation
import pickle
import os # Return File Names from a directory

def lazy_read_parquet(path):
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

def lazydict_to_parquet(lazydict, drive_path):
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

def read_pickle_jar(path):
    """
    Read all pickle files in a folder and return a dictionary of objects.
    
    Args:
        path (str): Relative or absolute path to the folder containing pickle files.
        
    Returns:
        dict: A dictionary where each key is the filename (without extension) and the value is the loaded object.
    """
    pickle_rick = {}  # Dictionary to store the loaded objects
    for filename in os.listdir(path):
        if filename.endswith(".pkl"):  # Check for pickle files
            file_path = os.path.join(path, filename)  # Construct full file path
            with open(file_path, 'rb') as file:  # Open file for reading in binary mode
                pickle_rick[os.path.splitext(filename)[0]] = pickle.load(file)  # Deserialize and store in dictionary
    return pickle_rick