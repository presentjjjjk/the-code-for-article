import numpy as np
import random
import os
import pickle
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Set random seed
def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

set_random_seed(42)

# Load dataset
file_path = 'dataset_T_b.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Prepare features and labels
X = data[[
    'Dipole Moment (Debye)', 'Polarizability', 'free_energy',
    'Chi0v', 'Chi0n', 'Chi1v', 'Chi1n', 'Chi2v', 'Chi2n',
    'Chi3v', 'Chi3n', 'Chi4v', 'Chi4n', 'Kappa1', 'Kappa2',
    'HallKierAlpha', 'BalabanJ', 'TPSA', 'LabuteASA',
    'SMR_VSA1', 'SMR_VSA9', 'PEOE_VSA1', 'PEOE_VSA14',
    'HBD', 'HBA', 'Num Rotatable Bonds', 'Num Aromatic Atoms',
    'Num Aromatic Rings', 'Num Aromatic Bonds', 'BertzCT',
    'Volume', 'Sphericity', 'Min_value', 'Max_value',
    'Overall_surface_area', 'Positive_surface_area',
    'Negative_surface_area', 'Overall_average',
    'Positive_average', 'Negative_average', 'Overall_variance',
    'Positive_variance', 'Negative_variance', 'Balance',
    'Internal_charge_separation', 'MPI', 'Nonpolar_surface_area',
    'Polar_surface_area', 'Overall_skewness', 'Positive_skewness',
    'Negative_skewness', 'LogP', 'Molecular Weight'
]]
y = data['boiling point']

# Ten-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store evaluation metrics
r2_scores = []
mae_scores = []
rmse_scores = []
max_error_scores = []
mre_scores = []

best_r2 = -float('inf')
best_model = None
best_scaler = None

# Function to add Gaussian noise
def add_gaussian_noise(X, noise_factor=0.02):
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

# Set SVR model hyperparameter search space
param_grid = {
    'C': np.linspace(0, 5000, 500),  # Range for parameter C
    'epsilon': [0.1, 0.2, 0.5],      # Choices for epsilon
    'kernel': ['rbf'],               # Use RBF kernel
    'gamma': ['scale']               # Use default gamma
}

# Initialize SVR model
svr = SVR()

# Use GridSearchCV for hyperparameter search
grid_search = GridSearchCV(svr, param_grid, scoring='r2', cv=3, verbose=1, n_jobs=-1)

# Training and evaluation in each fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Data standardization
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Further split test set into validation set
    X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.5, random_state=fold
    )

    # Data augmentation (add noise)
    X_train_noisy = add_gaussian_noise(X_train_scaled)
    X_train_augmented = np.vstack([X_train_scaled, X_train_noisy])
    y_train_augmented = np.concatenate([y_train, y_train])

    # Hyperparameter tuning on validation set
    grid_search.fit(X_train_augmented, y_train_augmented)

    # Output best parameters
    print(f"Best parameters found in fold {fold}: ", grid_search.best_params_)

    # Select best model
    best_model = grid_search.best_estimator_

    # Prediction and evaluation
    y_pred = best_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # Store evaluation results
    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)

    # Save best model
    if r2 > best_r2:
        best_r2 = r2

print('\n')
print(f"Best R² Score: {best_r2:.4f}")

# Output average performance metrics
print(f"Average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"Average Mean Relative Error: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")
