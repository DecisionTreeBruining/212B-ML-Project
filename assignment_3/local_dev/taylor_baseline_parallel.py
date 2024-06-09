import os
import time
import pickle
import logging
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score
import pandas as pd
import polars as pl
from assignment_3_tools import parquet_to_dict

# Unique Datasets.
def unq_df_names(lazy_dict):
    """
    Creates a set of unique datasets from a LazyFrame dictionary.
    ---
    Args: 
        lazy_dict (dict): Contains LazyFrame names and corresponding LazyFrames.
    Returns:
        unq_names (set): Contains unique dataset names.
    """
    all_names = list()
    for key in lazy_dict:
        if key[-6:] == "_train":
            all_names.append(key[:-8]) # Remove _X_train and _y_train
        elif key[-5:] == "_test":
            all_names.append(key[:-7]) # Remove _X_test and _y_test
        else:
            pass
    unq_names = set(all_names)
    return unq_names

# Return Corresponding Test Set.
def corr_testset(unq_name):
    """
    Return the names of testsets corresponding to a preprocessed trainset
    ---
    Args:
        unq_name(set): Contains unique dataset names.
    Returns:
        X_test_name(str): Name of corresponding predictor testset.
        y_test_name(str): Name of corresponding response testset.
    """
    threshold = unq_name[-2:] # 2 possibilities: "##" or "mp"
    if threshold.isnumeric():
        # Use null-threshold datasets with no balancing operations.
        X_test_name = f"df_heart_drop_{threshold}_imp_X_test"
        y_test_name = f"df_heart_drop_{threshold}_imp_y_test"
    else:
        # Use null-threshold datasets with no balancing operations. 
        X_test_name = f"{unq_name}_X_test"
        y_test_name = f"{unq_name}_y_test"
    return X_test_name, y_test_name

def process_dataset(name, lazy_dict, param_grid, save_pth):
    X_train_name = f"{name}_X_train"
    y_train_name = f"{name}_y_train"
    (X_test_name, y_test_name) = corr_testset(name)

    X_train = lazy_dict[X_train_name].collect().to_pandas()
    y_train = lazy_dict[y_train_name].collect().to_pandas()
    X_test = lazy_dict[X_test_name].collect().to_pandas()
    y_test = lazy_dict[y_test_name].collect().to_pandas()

    X_train.drop(columns=['__index_level_0__'], errors='ignore', inplace=True)
    y_train.drop(columns=['__index_level_0__'], errors='ignore', inplace=True)
    X_test.drop(columns=['__index_level_0__'], errors='ignore', inplace=True)
    y_test.drop(columns=['__index_level_0__'], errors='ignore', inplace=True)

    y_train = y_train.to_numpy().ravel()
    y_test = y_test.to_numpy().ravel()

    # Data scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Setup model and GridSearchCV
    mlp_model = MLPClassifier(early_stopping=True, verbose=False, random_state=212)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=212)
    grid_search = GridSearchCV(mlp_model, param_grid=param_grid, cv=cv, scoring='recall', verbose=1)
    
    
    logging.info(f"Processing dataset: {name}")
    print(f"Training on {name}...", flush=True)

    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train)
    fit_time = time.time() - start_time

    print(f"{name}GridSearch completed", flush=True)
    logging.info(f"{name}GridSearch completed.")    
    
    best_model = grid_search.best_estimator_
    
    y_pred_test = best_model.predict(X_test_scaled)
    test_recall = recall_score(y_test, y_pred_test)
    test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Save the model
    with open(f"{save_pth}{name}_MLPbaseline.pkl", 'wb') as file:
        pickle.dump(best_model, file)

    logging.info(f"{name}Model saved to {save_pth}")

    return {
        "Dataset Name": name,
        "Best Recall": test_recall,
        "Best ROCAUC": test_roc_auc,
        "Best Accuracy": test_accuracy,
        "Fit Time": fit_time}

def mlp_baseline(lazy_dict, unq_names, param_grid, save_pth, threads=None):
    if threads is None:
        threads = os.cpu_count() - 2  # Save some resources for other processes
        print(f"Using {threads} CPU threads!")
        
    results = []
    with ProcessPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process_dataset, name, lazy_dict, param_grid, save_pth) for name in unq_names]
        for future in futures:
            result = future.result()
            results.append(result)
            logging.info(f"Completed processing for {result['Dataset Name']}")
            print(f"Completed processing for {result['Dataset Name']}")
    
    results_df = pd.DataFrame(results)
    results_df.to_parquet("../../Data/GoogleDrive/baseline_results.parquet")
    return results_df

# Log Initialization
logging.basicConfig(filename='./log/MLP_baseline.log', filemode='w', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
## Paths
data_pth = "../../Data/GoogleDrive/Encoded_Data/"
save_pth = "../../Data/GoogleDrive/Baseline/"

## Read in Parquet to LazyFrame Dictionary.
pq_jar = parquet_to_dict(data_pth)

## Record the unique dataset names.
unq_names = unq_df_names(pq_jar)

## List the parameters. Keep this out of the loop for gridsearching later.
param_grid = {
    'hidden_layer_sizes': [(100,)],  # Single layer with 100 neurons
    'activation': ['relu'],  # Using 'relu' activation function
    'solver': ['adam'],  # Solver set to 'adam'
    'alpha': [0.0001],  # L2 penalty (regularization term)
    'batch_size': ['auto'],  # 'auto' sets batch size to min(200, n_samples)
    'learning_rate': ['constant'],  # Learning rate schedule
    'learning_rate_init': [0.001],  # Initial learning rate
    'power_t': [0.5],  # The exponent for inverse scaling learning rate
    'max_iter': [200],  # Maximum number of iterations
    'shuffle': [True],  # Whether to shuffle samples in each iteration
    'random_state': [212],  # Random state for reproducibility, can set to a specific number
    'tol': [0.0001],  # Tolerance for the optimization
    'verbose': [False],  # Whether to print progress messages to stdout
    'warm_start': [False],  # Reuse solution of the previous call to fit as initialization
    'momentum': [0.9],  # Momentum for gradient descent update
    'nesterovs_momentum': [True],  # Whether to use Nesterov's momentum
    'early_stopping': [True],  # Whether to use early stopping to terminate training
    'validation_fraction': [0.1],  # Proportion of training data to set aside as validation set
    'beta_1': [0.9],  # Exponential decay rate for estimates of first moment vector in adam
    'beta_2': [0.999],  # Exponential decay rate for estimates of second moment vector in adam
    'epsilon': [1e-08],  # Value for numerical stability in adam
    'n_iter_no_change': [10],  # Maximum number of epochs to not meet improvement threshold
    'max_fun': [15000]  # Maximum number of loss function calls
}

mlp_baseline(pq_jar, unq_names, param_grid, save_pth)