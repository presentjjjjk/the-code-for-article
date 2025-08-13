import numpy as np
import xgboost as xgb
import random
import os
import pickle
import json
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Set random seed
def set_random_seed(seed_value=42):
    """
    Set all relevant random seeds to ensure reproducibility of results
    Args:
        seed_value: Random seed value
    """
    # 1. Set Python random seed
    random.seed(seed_value)
    
    # 2. Set numpy random seed
    np.random.seed(seed_value)
    
    # 4. Set Python environment variables
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Settings for GPU
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
# Call function to set random seed
set_random_seed(42)

# Load data
file_path = 'dataset_T_b.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Features and target variable (keeping original features unchanged)
X = data[[
    'Dipole Moment (Debye)', 
    'Polarizability',
    'free_energy',
    # Part 2: Topological Graph Descriptors
    'Chi0v',
    'Chi0n',
    'Chi1v',
    'Chi1n',
    'Chi2v',
    'Chi2n',
    'Chi3v',
    'Chi3n',
    'Chi4v',
    'Chi4n',
    'Kappa1','Kappa2',
    'HallKierAlpha',
    'BalabanJ',
    # Molecular Surface Area Descriptors
    'TPSA',
    'LabuteASA',
    'SMR_VSA1', # Molecular surface area with high negative charge
    'SMR_VSA9', # Molecular surface area with high positive charge
    'PEOE_VSA1',
    'PEOE_VSA14',
    # Hydrogen Bond Descriptors
    'HBD','HBA',
    # Structural Information Descriptors
    'Num Rotatable Bonds',
    'Num Aromatic Atoms','Num Aromatic Rings','Num Aromatic Bonds',
    'BertzCT',
    'Volume',
    'Sphericity',
    # Electrostatic Potential Related Descriptors
    "Min_value", 
    "Max_value",
    "Overall_surface_area",
    "Positive_surface_area",
    "Negative_surface_area",
    "Overall_average",
    "Positive_average",
    "Negative_average",
    "Overall_variance",
    "Positive_variance",
    "Negative_variance",
    "Balance",
    "Internal_charge_separation",
    "MPI",
    "Nonpolar_surface_area",
    "Polar_surface_area",
    "Overall_skewness",
    "Positive_skewness",
    "Negative_skewness",
    # Additional Descriptors - Octanol-Water Partition Coefficient, Molecular Weight
    'LogP',
    'Molecular Weight',
]]
y = data['boiling point']

# Initialize cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores = []
mae_scores = []
rmse_scores = []
max_error_scores = []
mre_scores = []

# XGBoost parameters configuration
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 2000,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'random_state': 42,
    'n_jobs': -1
}

# Cross-validation loop
best_r2 = -float('inf')
best_model = None
best_evals_result = None  # New: save evaluation results of the best model

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_temp = X.iloc[train_index], X.iloc[test_index]
    y_train, y_temp = y.iloc[train_index], y.iloc[test_index]

    # Split development set and test set
    X_dev, X_test, y_dev, y_test, dev_indices, test_indices = train_test_split(
        X_temp, y_temp, test_index, test_size=0.5, random_state=fold
    )

    from xgboost import XGBRegressor, callback

    # Initialize model
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    )

    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_dev, y_dev)],  # Add training set to evaluation set
        early_stopping_rounds=300,   # Early stopping
        verbose=True,
        eval_metric='rmse',          # Use RMSE as evaluation metric
    )

    # Prediction and evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Store results
    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)
    
    
    # Save the best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_evals_result = model.evals_result()  # Save evaluation results

# Output statistical results
print(f"Best R²: {best_r2:.4f}")
print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"Average MRE: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")

# New: Plot training curves for the best model
if best_evals_result is not None:
    plt.figure(figsize=(10, 6))
    
    # Extract training and validation RMSE values
    train_rmse = best_evals_result['validation_0']['rmse']
    val_rmse = best_evals_result['validation_1']['rmse']
        
    # Plot original curves
    plt.plot(train_rmse, label='Training RMSE', alpha=1, color='tab:blue')
    plt.plot(val_rmse, label='Validation RMSE', alpha=1, color='tab:orange')
    
    # Mark best iteration
    best_iter = best_model.best_iteration
    plt.axvline(best_iter, color='gray', linestyle='--', 
                label=f'Best Iteration ({best_iter})')
    
    # Chart decorations
    plt.xlabel('Boosting Rounds', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Best Model Training Progress with Early Stopping', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('best_model_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No training curves available.")





