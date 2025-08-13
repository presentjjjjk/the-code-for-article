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
from tensorflow.keras.models import load_model

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
file_path = 'dataset_Vc.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Features and target variable (keeping original features)
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
y = data['V_c']*1000

# Initialize cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
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
best_evals_result = None

# Get predictions from another model through flexible comments
# import pickle

# # Load pre-trained model and Scaler
# pretrained_model = load_model('best_model.h5')
# with open('best_scaler.pkl', 'rb') as f:
#     pretrained_scaler = pickle.load(f)

# # Feature scaling
# X_total_scaled = pretrained_scaler.transform(X)

# # Predict boiling point
# predicted_boiling_point = pretrained_model.predict(X_total_scaled)

# # Ensure predictions are 1-dimensional array
# if predicted_boiling_point.ndim > 1:
#     predicted_boiling_point = predicted_boiling_point.flatten()

# X0=X.copy()

# # Add predictions to feature matrix
# X0['predicted_boiling_point'] = predicted_boiling_point
# # Load pre-trained model and Scaler
# pretrained_model = load_model('best_model_Tc.h5')
# with open('best_scaler_Tc.pkl', 'rb') as f:
#     pretrained_scaler = pickle.load(f)

# # Feature scaling
# X_total_scaled = pretrained_scaler.transform(X0)

# # Predict critical temperature
# predicted_Tc = pretrained_model.predict(X_total_scaled)

# # Ensure predictions are 1-dimensional array
# if predicted_Tc.ndim > 1:
#     predicted_Tc = predicted_Tc.flatten()

# # Add predictions to feature matrix
# X['Tc'] = predicted_Tc

# # Load pre-trained model and Scaler
# pretrained_model = load_model('best_model_pc.h5')
# with open('best_scaler_pc.pkl', 'rb') as f:
#     pretrained_scaler = pickle.load(f)

# # Feature scaling
# X_total_scaled = pretrained_scaler.transform(X)

# # Predict critical temperature
# predicted_pc = pretrained_model.predict(X_total_scaled)

# # Ensure predictions are 1-dimensional array
# if predicted_pc.ndim > 1:
#     predicted_pc = predicted_Tc.flatten()

# # Add predictions to feature matrix
# X['pc'] = predicted_pc

for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_temp = X.iloc[train_index], X.iloc[test_index]
    y_train, y_temp = y.iloc[train_index], y.iloc[test_index]

    # Split into development and test sets
    X_dev, X_test, y_dev, y_test, dev_indices, test_indices = train_test_split(
        X_temp, y_temp, test_index, test_size=0.5, random_state=fold
    )

    from xgboost import XGBRegressor, callback

    # Initialize model
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=2000,
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

    # Predict and evaluate
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
        
    # Save best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_evals_result = model.evals_result()  # Save evaluation results


import json
import pickle
# Save model to file
best_model.save_model('best_model_Vc.model')

# Output statistical results
print(f"Best R²: {best_r2:.4f}")
print(f"Average R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"Average MRE: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")

# Use the best model to predict on the entire dataset
final_predictions = best_model.predict(X).flatten()
final_actuals = y.values

# Visualize the comparison between predicted and actual values
plt.figure(figsize=(8, 6))

# Set chart style
plt.style.use('default')  # Switch to default style

# Calculate absolute errors
abs_errors = np.abs(final_predictions - final_actuals)

# Create gradient scatter plot
scatter = plt.scatter(final_actuals, final_predictions, 
                     alpha=0.8,                          
                     s=30,                              
                     c=abs_errors,                   # Change to absolute errors
                     cmap='ocean',                 # Use YlOrRd colormap (yellow to red)
                     edgecolor='white',                 
                     linewidth=0.5)                     

# Add ideal prediction line
min_val = min(final_actuals.min(), final_predictions.min())
max_val = max(final_actuals.max(), final_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 
         color='#4169E1',                              # Changed to royal blue
         linestyle='-',                               
         linewidth=1.5,                                
         label="ideal line")

# Optimize labels and title
plt.xlabel('Actual Vc (cm^3/mol)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Vc (cm^3/mol)', fontsize=12, fontweight='bold')
plt.title('Xgboost Model Prediction vs Actual Vc(Entire Dataset)', fontsize=14, pad=15)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Absolute Error (cm^3/mol)', fontsize=10)  # Changed label

# Optimize grid lines
plt.grid(True, linestyle='--', alpha=0.3)

# Adjust margins
plt.tight_layout()
# Add the legend and save the plot
plt.legend(loc='upper left')
plt.savefig('Actual_vs_Predicted_Vc_Enhanced.png', dpi=300, bbox_inches='tight')
plt.show()

# New: Plot training curve for the best model
if best_evals_result is not None:
    plt.figure(figsize=(8, 6))
    
    # Extract training and validation RMSE values
    train_rmse = best_evals_result['validation_0']['rmse']
    val_rmse = best_evals_result['validation_1']['rmse']
        
    # Plot original curve
    plt.plot(train_rmse, label='Training RMSE', alpha=1, color='tab:blue')
    plt.plot(val_rmse, label='Validation RMSE', alpha=1, color='tab:orange')
    
    # Mark best iteration
    best_iter = best_model.best_iteration
    plt.axvline(best_iter, color='gray', linestyle='--', 
                label=f'Best Iteration ({best_iter})')
    
    # Chart decoration
    plt.xlabel('Boosting Rounds', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title('Best Model Training Progress with Early Stopping', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('best_model_training_curve(Vc).png', dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("No training curves available.")





