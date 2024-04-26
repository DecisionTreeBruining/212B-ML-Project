import polars as pl # Lazy Dataframe Manipulation 
import pandas as pd # Dataframe
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