import numpy as np
import tensorflow as tf
import random
import os
import pickle
import json
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

    # GPU settings
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

# set random seed
set_random_seed(42)

# Load the dataset
file_path = 'dataset_T_b.csv'
# file_path = 'dataset_Tc.csv'
data = pd.read_csv(file_path,encoding='latin1')

# Features
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
    # Additional Descriptors - LogP, Molecular Weight
    'LogP',
    'Molecular Weight',
]]
y = data['boiling point']

batch_size = 512

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

# In the cross-validation loop, find the best model and save it
best_r2 = -float('inf')
best_model = None
best_scaler = None
fold_histories = []  # Store training history for each fold


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

    # Data augmentation: Add samples with noise
    X_train_noisy = add_gaussian_noise(X_train_scaled)
    X_train_augmented = X_train_scaled #np.vstack([X_train_scaled, X_train_noisy])
    y_train_augmented = y_train # np.concatenate([y_train, y_train])

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
        X_train_augmented, y_train_augmented, 
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
    mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100 

   
    fold_histories.append([history,r2])

    # Store the metrics for this fold
    r2_scores.append(r2)
    mae_scores.append(mae)
    rmse_scores.append(rmse)
    max_error_scores.append(max_error)
    mre_scores.append(mre)

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_scaler = scaler

# Save the best model and scaler
best_model.save('best_model.h5')  # Save model structure and weights
with open('best_scaler.pkl', 'wb') as f:
    pickle.dump(best_scaler, f)  # Save scaler

# Save model configuration (optional)
model_config = best_model.get_config()
with open('model_config.json', 'w') as f:
    json.dump(model_config, f)


for i in range(len(fold_histories)):
    if fold_histories[i][1]==best_r2:
        best_history = fold_histories[i][0]
        break

# Plot loss curve for the best model
plt.figure(figsize=(8, 6))

# Get training and validation loss
train_loss = best_history.history['loss']
val_loss = best_history.history['val_loss']

# Use log scale
plt.yscale('log')

# Plot curves
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Development Loss')

# Set Y axis range to avoid extreme values
plt.ylim(bottom=min(train_loss + val_loss) * 0.9, top=max(train_loss + val_loss) * 1.1)

# Add labels, title and legend
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE, log scale)')
plt.title('Best Model - Loss Curve (Log Scale)')
plt.legend()
plt.grid(True)

# Save and display the plot
plt.savefig('Best_Model_Loss_Curve.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Best model saved: R² = {best_r2:.4f})")

# Calculate and display the mean and standard deviation of R² and MAE
print(f"Average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
print(f"Average Mean Relative Error: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")

# Load the entire dataset using the best scaler
X_all_scaled = best_scaler.transform(X)

# Use the best model to predict on the entire dataset
final_predictions = best_model.predict(X_all_scaled).flatten()
final_actuals = y.values

# Calculate absolute errors
abs_errors = np.abs(final_predictions - final_actuals)

# Visualize the comparison between predicted and actual values
plt.figure(figsize=(8, 6))

# Set chart style
plt.style.use('default')  # Switch to default style

# Create gradient scatter plot
scatter = plt.scatter(final_actuals, final_predictions, 
                     alpha=0.8,                          
                     s=30,                              
                     c=abs_errors,                   # Change to absolute errors
                     cmap='ocean',                    
                     edgecolor='white',                 
                     linewidth=0.5)                     

# Add ideal prediction line
min_val = min(final_actuals.min(), final_predictions.min())
max_val = max(final_actuals.max(), final_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val], 
         color='#FF6B6B',                              
         linestyle='-',                               
         linewidth=1.5,                                
         label="ideal line")

# Optimize labels and title
plt.xlabel('Actual Boiling Point (K)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Boiling Point (K)', fontsize=12, fontweight='bold')
plt.title('Neural Networks Model Prediction vs Actual Boiling Points(Entire Dataset)', fontsize=14, pad=15)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Absolute Error (K)', fontsize=10)  # Changed label

# Optimize grid lines
plt.grid(True, linestyle='--', alpha=0.3)

# Adjust margins
plt.tight_layout()
# Add the legend and save the plot
plt.legend(loc='upper left')
plt.savefig('Actual_vs_Predicted_Boiling_Points_Enhanced.png', dpi=300, bbox_inches='tight')
plt.show()
