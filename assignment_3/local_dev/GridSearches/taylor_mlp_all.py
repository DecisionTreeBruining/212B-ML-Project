import os
import time # Runtime
import pickle # Model Saving
import logging # Log Checkpoints
import numpy as np # Flatten y vectors
import pandas as pd # DataFrame
import polars as pl # LazyFrame
from sklearn.preprocessing import StandardScaler # X Standardization
from sklearn.neural_network import MLPClassifier as mlp # model
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score # Scoring
from sklearn.model_selection import GridSearchCV, StratifiedKFold, ParameterGrid
from great_tables import GT, md, html, from_column, style, loc, vals
from assignment_3_tools import parquet_to_dict, unq_df_names, corr_testset

def mlp_gridsearch(lazy_dict, unq_names, param_grid, save_pth, test_name, threads=None):
    """
    MLP GridSearch using 5-fold Cross Validation. Saves best model and results.
    ---
    Args:
        lazy_dict: Dictionary with names and LazyFrames of train and test sets.
        unq_names: List of unique names of parent datasets.
        param_grid: Dictionary of parameters for MLPClassifier.
        save_pth: String of the path to save the best model.
        test_name: String of the test performed.
        threads: Integer of CPU threads for cross-validation (optional).
    Return:
        None
    """
    ## Initializing
    # Define number of threads to be used in GridSearch
    if threads is None:
        threads = os.cpu_count()
        print(f"Using {threads} CPU threads!")

    # Log for debugging
    logging.basicConfig(
        filename=f"./log/MLP_{test_name}.log",
        filemode='w', 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    ## GridSearch and Results
    for name in unq_names:
        # Results from prediction on test_set
        best_results = {
            "Dataset_Name": [],
            "Grid_Variable": [],
            "Parameters": [],
            "Recall": [], 
            "ROC_AUC": [], 
            "Accuracy": [],
            "Fit_Time": []}
        
        # Results from prediction on Cross Validation Set
        param_results = {
            "Dataset_Name": [],
            "Grid_Variable": [],
            "Parameters": [],
            "Recall": [], 
            "Fit_Time": []}
        
        ## Reading and Preparing Data
        # Dataset names in path
        X_train_name = f"{name}_X_train"
        y_train_name = f"{name}_y_train"
        X_test_name = f"{name}_X_test"
        y_test_name = f"{name}_y_test"

        # Train and test sets.
        X_train = lazy_dict[X_train_name].collect().to_pandas()
        y_train = lazy_dict[y_train_name].collect().to_pandas()
        X_test = lazy_dict[X_test_name].collect().to_pandas()
        y_test = lazy_dict[y_test_name].collect().to_pandas()

        # Drop index column
        X_train.drop(columns=['__index_level_0__'], errors='ignore', inplace=True)
        y_train.drop(columns=['__index_level_0__'], errors='ignore', inplace=True)
        X_test.drop(columns=['__index_level_0__'], errors='ignore', inplace=True)
        y_test.drop(columns=['__index_level_0__'], errors='ignore', inplace=True)

        # Flatten response sets
        y_train = y_train.to_numpy().ravel()
        y_test = y_test.to_numpy().ravel()

        # Standardize predictor sets
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ## Defining Modeling and GridSearch
        # Define cross-validation folds
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=212)

        # Define GridSearch
        grid_search = GridSearchCV(
            mlp(),
            param_grid=param_grid, 
            cv=cv,
            scoring='recall',
            n_jobs=threads,
            verbose=1, 
            return_train_score=True)


        ## Performing GridSearch
        # Debugging Checkpoint
        logging.info(f"Processing dataset: {name}")
        print(f"Training on {name}...", flush=True)

        # GridSearch Training and Results
        grid_search.fit(X_train_scaled, y_train)

        # Debugging Checkpoint
        print(f"GridSearch completed", flush=True)
        logging.info(f"GridSearch for {test_name} completed.")

        ## Results from GridSearch
        # Storing Results for each parameter combination
        for i in range(len(grid_search.cv_results_['params'])):
            param_combination = grid_search.cv_results_['params'][i]
            recall = grid_search.cv_results_['mean_test_score'][i]
            fit_time = grid_search.cv_results_['mean_fit_time'][i]
            param_results["Dataset_Name"].append(name)
            param_results["Grid_Variable"].append(test_name)
            param_results["Parameters"].append(param_combination)
            param_results["Recall"].append(recall)
            param_results["Fit_Time"].append(fit_time)
        
        # Convert to DataFrame
        param_results_df = pd.DataFrame(param_results)
        param_results_df = param_results_df.sort_values(by="Recall", ascending=False)

        # Best model by Recall on Cross Validation data
        best_fit_time = param_results_df.iloc[0]["Fit_Time"]
        best_model = grid_search.best_estimator_

        # Metrics on test set
        y_pred_test = best_model.predict(X_test_scaled)
        test_recall = recall_score(y_test, y_pred_test)
        test_roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
        test_accuracy = accuracy_score(y_test, y_pred_test)

        # Save best model as pickle
        with open(f"{save_pth}best_model{test_name}.pkl", 'wb') as file:
            pickle.dump(best_model, file)

        # Debugging Checkpoint
        logging.info(f"Model saved to {save_pth}")

        # Results from predicting test data using the best model.
        best_results["Dataset_Name"].append(name)
        best_results["Grid_Variable"].append(test_name)
        best_results["Parameters"].append(grid_search.best_params_)
        best_results["Recall"].append(test_recall)
        best_results["ROC_AUC"].append(test_roc_auc)
        best_results["Accuracy"].append(test_accuracy)
        best_results["Fit_Time"].append(best_fit_time)

        # Convert to DataFrame
        best_results_df = pd.DataFrame(best_results)

        # Save results as Parquet
        best_results_df.to_parquet(f"{save_pth}test_results{test_name}.parquet", index=False)
        param_results_df.to_parquet(f"{save_pth}grid_results{test_name}.parquet", index=False)

        # Debugging Checkpoint
        print(f"{test_name} GridSearch completed!", flush=True)
        logging.info(f"{test_name} GridSearch completed!")

# Train and test sets are in MLP_Dataset.
# Save results and best model to MLP_Results.
data_pth = "../../../Data/GoogleDrive/MLP_Dataset/"
save_pth = "../../../Data/GoogleDrive/MLP_Results/"

# Read in Parquet files in path and add to a LazyFrame dictionary.
pq_jar = parquet_to_dict(data_pth)

# Record the unique dataset names (drop X_train, etc.).
unq_names = unq_df_names(pq_jar)

# all parameters
all_test_parameters = {
    '-baseline': {
        'hidden_layer_sizes': [(100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.0001],
        'batch_size': ['auto'],
        'learning_rate': ['constant'],
        'learning_rate_init': [0.001],
        'max_iter': [200],
        'momentum': [0.9],
        'n_iter_no_change': [10]},
    'neurons-hidden_layer_sizes': {'hidden_layer_sizes': [(1), (50), (250), (500)]},
    'layers-hidden_layer_sizes': {'hidden_layer_sizes': [(100, 100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100)]},
    '-activation': {'activation': ['identity', 'logistic', 'tanh']},
    '-batch_size': {'batch_size': [1, 100, 500, 1000]},
    '-learning_rate': {'learning_rate': ['invscaling', 'adaptive']},
    '-learning_rate_init': {'learning_rate_init': [0.0001, 0.01, .1]},
    '-max_iter': {'max_iter': [100, 250, 500, 1000]},
    '-random_state': {'ramdom_state': [100, 101, 102]},
    '-solver': {'solver': ['sgd', 'lbfgs']},
    '-alpha': {'alpha': [0.0, 0.25, 0.5, 0.75, 1.0]},
    '-momentum': {'momentum': [0.0, 0.25, 0.5, 0.75, 1.0]},
    '-n_iter_no_change': {'n_iter_no_change': [50, 100, 250, 500]}
}


# Run the model
for key, value in all_test_parameters.items():
    mlp_gridsearch(pq_jar, unq_names, value, save_pth, key, 16)