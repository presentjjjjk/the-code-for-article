from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout,Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,max_error
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

# Call function to set random seed
set_random_seed(42)

def build_boiling_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # First layer: 128 neurons, L2 regularization, batch normalization, ReLU activation, Dropout
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Second layer: 64 neurons, L2 regularization, batch normalization, ReLU activation, Dropout
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    # Third layer: 32 neurons, L2 regularization, batch normalization, ReLU activation, Dropout
    x = Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    
    # Fourth layer: 16 neurons, L2 regularization, batch normalization, ReLU activation
    x = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output layer: 1 neuron, linear activation
    boiling_point = Dense(1, activation='linear', name='boiling_point')(x)
    
    return Model(inputs, boiling_point, name='boiling_model')

def build_critical_model(input_shape, boiling_dim=1):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 沸点输入（用于残差连接）
    boiling_input = Input(shape=(boiling_dim,), name='boiling_input')
    
    # First layer: 128 neurons, L2 regularization, batch normalization, ReLU activation, Dropout
    x = Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Second layer: 64 neurons, L2 regularization, batch normalization, ReLU activation, Dropout
    x = Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    
    # Third layer: 32 neurons, L2 regularization, batch normalization, ReLU activation, Dropout
    x = Dense(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)
    
    # Fourth layer: 16 neurons, L2 regularization, batch normalization, ReLU activation (倒数第二层)
    x = Dense(16, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 残差连接：将沸点输出映射到16维并与倒数第二层输出相加
    boiling_mapped = Dense(16, activation='linear', name='boiling_mapper')(boiling_input)
    x_residual = tf.keras.layers.Add()([x, boiling_mapped])
    
    # Output layer: 1 neuron, linear activation
    critical_temp = Dense(1, activation='linear', name='critical_temp')(x_residual)
    
    return Model(inputs=[inputs, boiling_input], outputs=critical_temp, name='critical_model')

# Define joint model
# Ensure output layer names match loss dictionary keys
def build_joint_model(boiling_model, critical_model, X):
    input_shape = X.shape[1]
    input_layer = tf.keras.Input(shape=(input_shape,), name='input_layer')
    
    # Define boiling model output and specify name when defining model
    boiling_output = boiling_model(input_layer)
    boiling_output = tf.keras.layers.Lambda(lambda x: x, name='boiling_point')(boiling_output)
    
    # Critical temperature model with residual connection
    # 将原始输入和沸点输出都传给临界温度模型
    critical_output = critical_model([input_layer, boiling_output])
    critical_output = tf.keras.layers.Lambda(lambda x: x, name='critical_temp')(critical_output)
    
    # Define joint model
    joint_model = tf.keras.Model(inputs=input_layer, outputs=[boiling_output, critical_output], name='joint_model')
    return joint_model


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

# Load the dataset
file_path = 'dataset_Tc.csv'
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
    'SMR_VSA1', # Molecules with higher negative surface area
    'SMR_VSA9', # Molecules with higher positive surface area
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
    # Supplemental Descriptor-Hydrophobicity, Molecular Weight
    'LogP',
    'Molecular Weight',
    # 'boiling point'
]]

y_boiling =data['boiling point']
y_critical = data['T_c']

# Instantiate model
# Instantiate model
boiling_model = build_boiling_model(input_shape=X.shape[1])
critical_model = build_critical_model(input_shape=X.shape[1], boiling_dim=1)  # 修改这里

import pickle
# Load pre-trained boiling model weights and scaler
pretrained_model = load_model('best_model.h5')  # Load pre-trained model
pretrained_model_weights = pretrained_model.get_weights()  # Get weights
with open('best_scaler.pkl', 'rb') as f:
    pretrained_scaler = pickle.load(f)

# Load pre-trained weights to boiling model
boiling_model.set_weights(pretrained_model_weights)

# If you need to freeze boiling model weights
for layer in boiling_model.layers[:-2]:
    layer.trainable = False


from sklearn.model_selection import KFold
import numpy as np

# Define 10-fold cross validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

r2_critical_scores = []
mae_critical_scores = []
rmse_critical_scores = []
max_error_critical_scores = []  
mre_critical_scores = []       

# In cross validation loop, find best model and save
best_r2 = -float('inf')
best_model = None
best_scaler = None
fold_histories = []  # For saving training history for each fold

# 10-fold cross validation loop
for fold, (train_idx, temp_idx) in enumerate(kfold.split(X)):
    print(f"Fold {fold + 1}")

    # Split training set, development set and test set
    X_train_fold, X_temp_fold = X.iloc[train_idx], X.iloc[temp_idx]
    y_boiling_train_fold, y_boiling_temp_fold = y_boiling.iloc[train_idx], y_boiling.iloc[temp_idx]
    y_critical_train_fold, y_critical_temp_fold = y_critical.iloc[train_idx], y_critical.iloc[temp_idx]

    # Split development and test sets
    X_dev, X_test, y_boiling_dev, y_boiling_test,y_critical_dev,y_critical_test = train_test_split(
        X_temp_fold, y_boiling_temp_fold,y_critical_temp_fold, test_size=0.5, random_state=fold
    )

    # Use pre-trained scaler for normalization
    X_train_fold_scaled = pretrained_scaler.transform(X_train_fold)
    X_dev_scaled = pretrained_scaler.transform(X_dev)
    X_test_scaled = pretrained_scaler.transform(X_test)

    # Data augmentation: Add samples with noise
    X_train_fold_noisy = add_gaussian_noise(X_train_fold_scaled)
    X_train_fold_augmented = np.vstack([X_train_fold_scaled, X_train_fold_noisy])
    y_boiling_train_fold_augmented = np.concatenate([y_boiling_train_fold, y_boiling_train_fold])
    y_critical_train_fold_augmented = np.concatenate([y_critical_train_fold, y_critical_train_fold])

    # Build joint model
    joint_model = build_joint_model(boiling_model, critical_model, X)

    optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.1, 
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

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)

    # Compile model
    joint_model.compile(
        optimizer=optimizer,
        loss={
            'boiling_point': 'mse',  # Boiling model loss
            'critical_temp': 'mse'   # Critical temperature model loss
        },
        loss_weights={
            'boiling_point': 0.1,
            'critical_temp': 0.9
        }
    )

    # Train model
    history = joint_model.fit(
        X_train_fold_augmented,
        {'boiling_point': y_boiling_train_fold_augmented, 'critical_temp': y_critical_train_fold_augmented},
        epochs=2000,
        batch_size=128,
        verbose=1,
        validation_data=(X_dev_scaled, {'boiling_point': y_boiling_dev, 'critical_temp': y_critical_dev}),
        callbacks=[early_stopping, lr_scheduler]
    )


    # Use joint model for prediction
    y_pred_boiling, y_pred_critical = joint_model.predict(X_test_scaled)


    y_critical_test = y_critical_test.to_numpy().flatten()

    y_pred_critical = y_pred_critical.flatten()

    # Calculate critical temperature R², MAE, RMSE, max_error and MRE
    r2_critical = r2_score(y_critical_test, y_pred_critical)
    mae_critical = mean_absolute_error(y_critical_test, y_pred_critical)
    rmse_critical = np.sqrt(mean_squared_error(y_critical_test, y_pred_critical))
    max_error_critical = max_error(y_critical_test, y_pred_critical)
    mre_critical = np.mean(np.abs((y_critical_test - y_pred_critical) / (y_critical_test + 1e-9)))  # Avoid division by 0

    print(f"Critical Temperature Metrics for Fold {fold + 1}:")
    print(f"R²: {r2_critical:.4f}")
    print(f"MAE: {mae_critical:.4f}")
    print(f"RMSE: {rmse_critical:.4f}\n")

    # Save training history for each fold in fold_histories
    fold_histories.append([history,r2_critical])

    r2_critical_scores.append(r2_critical)
    mae_critical_scores.append(mae_critical)
    rmse_critical_scores.append(rmse_critical)
    max_error_critical_scores.append(max_error_critical)
    mre_critical_scores.append(mre_critical)



    # Save best model
    if r2_critical > best_r2:
        best_r2 = r2_critical
        best_model = joint_model

for i in range(len(fold_histories)):
    if fold_histories[i][1]==best_r2:
        best_history = fold_histories[i][0]
        break

# Plot loss curve for best model
plt.figure(figsize=(8, 6))

# Get training and validation loss
train_loss = best_history.history['loss']
val_loss = best_history.history['val_loss']

# Use logarithmic scale
plt.yscale('log')

# Plot curve
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Development Loss')

# Set Y axis range, avoid extreme values affecting
plt.ylim(bottom=min(train_loss + val_loss) * 0.9, top=max(train_loss + val_loss) * 1.1)

# Add labels, title and legend
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE, log scale)')
plt.title('Best Model - Loss Curve (Log Scale)')
plt.legend()
plt.grid(True)

# Save image and display
plt.savefig('Best_Model_Loss_Curve(Tc).png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Best model saved: R² = {best_r2:.4f}")

avg_r2_critical = np.mean(r2_critical_scores)
avg_mae_critical = np.mean(mae_critical_scores)
avg_rmse_critical = np.mean(rmse_critical_scores)


print("Average Critical Temperature Metrics across 10 Folds:")
print(f"R²: {avg_r2_critical:.4f}")
print(f"MAE: {avg_mae_critical:.4f}")
print(f"RMSE: {avg_rmse_critical:.4f}")
print(f'max_error:{np.mean(max_error_critical_scores):.4f}')
print(f'MRE: {np.mean(mre_critical_scores):.4f}')


# Load the entire dataset using the best scaler
X_all_scaled = pretrained_scaler.transform(X)

# Use best model to predict entire dataset
boiling_predictions, critical_predictions = best_model.predict(X_all_scaled)

# Extract critical temperature predictions
final_predictions = critical_predictions.flatten()
final_actuals = y_critical.values.flatten()

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
plt.xlabel('Actual Tc (K)', fontsize=12, fontweight='bold')
plt.ylabel('Predicted Tc (K)', fontsize=12, fontweight='bold')
plt.title('Neural Networks Model Prediction vs Actual Tc(Entire Dataset)', fontsize=14, pad=15)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Absolute Error (K)', fontsize=10)  # Changed label

# Optimize grid lines
plt.grid(True, linestyle='--', alpha=0.3)

# Adjust margins
plt.tight_layout()
# Add the legend and save the plot
plt.legend(loc='upper left')
plt.savefig('Actual_vs_Predicted_Tc_Enhanced.png', dpi=300, bbox_inches='tight')
plt.show()
