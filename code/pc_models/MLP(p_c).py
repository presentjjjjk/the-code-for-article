import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
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
    
    # 3. Set tensorflow random seed
    tf.random.set_seed(seed_value)
    
    # 4. Set Python environment variables
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    # Settings for GPU
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    # 5. Configure tensorflow settings
    tf_config = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1
    )
    tf.compat.v1.set_random_seed(seed_value)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=tf_config)
    tf.compat.v1.keras.backend.set_session(sess)

# Call function to set random seed
set_random_seed(42)

# Load the dataset
file_path = 'dataset_pc.csv'
data = pd.read_csv(file_path,encoding='latin1')

# Prepare the dataset
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
    'Molecular Weight'
]]
y = data['p_c']

# Get other two models through flexible comments
import pickle

# Load pre-trained model and Scaler
pretrained_model = load_model('best_model.h5')
with open('best_scaler.pkl', 'rb') as f:
    pretrained_scaler = pickle.load(f)

# Feature scaling
X_total_scaled = pretrained_scaler.transform(X)

# Predict boiling point
predicted_boiling_point = pretrained_model.predict(X_total_scaled)

# Ensure predictions are 1-dimensional array
if predicted_boiling_point.ndim > 1:
    predicted_boiling_point = predicted_boiling_point.flatten()

X0=X.copy()

# Add predictions to feature matrix
X0['predicted_boiling_point'] = predicted_boiling_point

# Load pre-trained model and Scaler
pretrained_model = load_model('best_model_Tc.h5')
with open('best_scaler_Tc.pkl', 'rb') as f:
    pretrained_scaler = pickle.load(f)

# Feature scaling
X_total_scaled = pretrained_scaler.transform(X0)

# Predict critical temperature
predicted_Tc = pretrained_model.predict(X_total_scaled)

# Ensure predictions are 1-dimensional array
if predicted_Tc.ndim > 1:
    predicted_Tc = predicted_Tc.flatten()

# Add predictions to feature matrix
X['Tc'] = predicted_Tc

batch_size = 128

# Initialize cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
r2_scores = []
mae_scores = []
rmse_scores = []
max_error_scores = []
mre_scores = []

# Function to add Gaussian noise
def add_gaussian_noise(X, noise_factor=0.02):
    """
    Add Gaussian noise to input data
    Args:
        X: Input data
        noise_factor: Noise intensity factor
    Returns:
        Data with added noise
    """
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

# Find the best model in cross-validation loop and save it
best_r2 = -float('inf')
best_model = None
best_scaler = None

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_temp = X.iloc[train_index], X.iloc[test_index]
    y_train, y_temp = y.iloc[train_index], y.iloc[test_index]

    X_dev, X_test, y_dev, y_test, dev_indices, test_indices = train_test_split(
        X_temp, y_temp, test_index, test_size=0.5, random_state=fold
    )

    # Normalize the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)

    # Data augmentation: add noisy samples
    X_train_noisy = add_gaussian_noise(X_train_scaled)
    X_train_augmented = np.vstack([X_train_scaled, X_train_noisy])
    y_train_augmented = np.concatenate([y_train, y_train])

    # Build the neural network model
    model = Sequential([
        Dense(128, input_dim=X_train.shape[1], kernel_regularizer=tf.keras.regularizers.l2(0.1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        
        Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.05)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.2),
        
        Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.1),
        
        Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dense(1),
        tf.keras.layers.Activation('linear')
    ])

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.5, 
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    # Add learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=20,
        min_lr=1e-9
    )

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # Set up EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)

    # Train the model with EarlyStopping on the development set
    history = model.fit(
        X_train_augmented, y_train_augmented,  # Use augmented data
        epochs=2000,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_dev_scaled, y_dev),
        callbacks=[early_stopping, lr_scheduler]
    )

    # Evaluate the model on the test fold
    y_pred = model.predict(X_test_scaled).flatten()
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Convert to percentage

    # Store the metrics for this fold
    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)

    # Save the best model
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_scaler = scaler

import json
import pickle
# Save best model and scaler
best_model.save('best_model_pc.h5')  # Save model structure and weights
with open('best_scaler_pc.pkl', 'wb') as f:
    pickle.dump(best_scaler, f)  # Save scaler

# Save model configuration
model_config = best_model.get_config()
with open('model_config_pc.json', 'w') as f:
    json.dump(model_config, f)

print(f"Best model: R² = {best_r2:.4f}")

# Calculate and display the mean and standard deviation of R² and MAE
print(f"Average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"Average Mean Relative Error: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")






