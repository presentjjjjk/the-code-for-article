import numpy as np
import random
import os
import pickle
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regression model

# Set random seed
def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
set_random_seed(42)

# Load data
file_path = 'dataset_T_b.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Features and target variable
X = data[[ 
    'Dipole Moment (Debye)', 'Polarizability', 'free_energy', 'Chi0v', 'Chi0n', 'Chi1v', 'Chi1n', 
    'Chi2v', 'Chi2n', 'Chi3v', 'Chi3n', 'Chi4v', 'Chi4n', 'Kappa1', 'Kappa2', 'HallKierAlpha', 
    'BalabanJ', 'TPSA', 'LabuteASA', 'SMR_VSA1', 'SMR_VSA9', 'PEOE_VSA1', 'PEOE_VSA14', 'HBD', 'HBA', 
    'Num Rotatable Bonds', 'Num Aromatic Atoms', 'Num Aromatic Rings', 'Num Aromatic Bonds', 'BertzCT', 
    'Volume', 'Sphericity', 'Min_value', 'Max_value', 'Overall_surface_area', 'Positive_surface_area', 
    'Negative_surface_area', 'Overall_average', 'Positive_average', 'Negative_average', 'Overall_variance', 
    'Positive_variance', 'Negative_variance', 'Balance', 'Internal_charge_separation', 'MPI', 
    'Nonpolar_surface_area', 'Polar_surface_area', 'Overall_skewness', 'Positive_skewness', 
    'Negative_skewness', 'LogP', 'Molecular Weight'
]]
y = data['boiling point']

# Initialize cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores, mae_scores, rmse_scores, max_error_scores, mre_scores = [], [], [], [], []

# Initialize Random Forest Regressor hyperparameters
rf_params = {
    'n_estimators': 10000,  # Number of trees
    'max_depth': 20,        # Maximum depth of trees, prevents overfitting
    'min_samples_split': 4, # Minimum samples for splitting, prevents overfitting
    'min_samples_leaf': 2,  # Minimum samples in leaf nodes, prevents overfitting
    'max_features': 'sqrt', # Number of features to consider at each split, reduces overfitting
    'random_state': 42,     # Random seed
    'n_jobs': -1           # Use all CPU cores
}

best_r2 = -float('inf')
best_model = None

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_temp = X.iloc[train_index], X.iloc[test_index]
    y_train, y_temp = y.iloc[train_index], y.iloc[test_index]

    # Initialize Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=rf_params['n_estimators'],
        max_depth=rf_params['max_depth'],
        min_samples_split=rf_params['min_samples_split'],
        min_samples_leaf=rf_params['min_samples_leaf'],
        max_features=rf_params['max_features'],
        random_state=rf_params['random_state'],
        n_jobs=rf_params['n_jobs'],
        verbose=1
    )

    # Train model
    model.fit(X_train, y_train)

    # Prediction and evaluation
    y_pred = model.predict(X_temp)
    r2 = r2_score(y_temp, y_pred)
    mae = mean_absolute_error(y_temp, y_pred)
    rmse = np.sqrt(mean_squared_error(y_temp, y_pred))
    max_error = np.max(np.abs(y_temp - y_pred))
    mre = np.mean(np.abs((y_temp - y_pred) / y_temp)) * 100

    # Store results
    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)

    # Save best model
    if r2 > best_r2:
        best_r2 = r2

# Output statistical results
print(f"Best R²: {best_r2:.4f}")
print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"Average MRE: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")

