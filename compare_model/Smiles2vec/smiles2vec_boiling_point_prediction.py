#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SMILES to Boiling Point Prediction using CNN with 10-fold Cross Validation
Based on the original smiles2vec.ipynb architecture and MLP validation strategy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import pickle
import json

# Keras/TensorFlow imports
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, Flatten, BatchNormalization, GRU, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    return data

def create_charset_and_encoding(smiles_list):
    # Create character set from all SMILES strings, add start and end tokens
    charset = set("".join(list(smiles_list)) + "!E")
    char_to_int = dict((c, i) for i, c in enumerate(charset))
    int_to_char = dict((i, c) for i, c in enumerate(charset))
    
    # Calculate maximum length for padding
    embed = max([len(smile) for smile in smiles_list]) + 5
    
    print(f"Character set: {charset}")
    print(f"Vocabulary size: {len(charset)}, Max length: {embed}")
    
    return charset, char_to_int, int_to_char, embed

def vectorize_smiles(smiles_array, char_to_int, embed, charset):
    """
    Convert SMILES strings to one-hot encoded vectors
    """
    one_hot = np.zeros((smiles_array.shape[0], embed, len(charset)), dtype=np.int8)
    
    for i, smile in enumerate(smiles_array):
        # Encode the start character
        one_hot[i, 0, char_to_int["!"]] = 1
        # Encode the rest of the characters
        for j, c in enumerate(smile):
            if j + 1 < embed - 1:  # Leave space for end character
                one_hot[i, j + 1, char_to_int[c]] = 1
        # Encode end character
        one_hot[i, len(smile) + 1:, char_to_int["E"]] = 1
    
    # Return input sequences (excluding last character for input)
    return one_hot[:, 0:-1, :]

def coeff_determination(y_true, y_pred):
    """
    Custom R² metric for Keras
    """
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))

def get_lr_metric(optimizer):
    """
    Learning rate metric for monitoring
    """
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

def create_model(vocab_size, embed_length):
    """
    Create the CNN model with the same architecture as the original
    """
    model = Sequential()
    
    # Embedding layer
    model.add(Embedding(vocab_size, 50, input_length=embed_length))
    
    # Convolutional layers
    model.add(Conv1D(192, 3, activation='relu'))
    
    # Bidirectional GRU layers
    model.add(Bidirectional(GRU(224, return_sequences=True)))
    model.add(Bidirectional(GRU(384, return_sequences=False)))
    
    model.add(Dense(1, activation='linear'))
    
    return model

def prepare_data_for_training(X_vectorized):
    """
    Convert one-hot encoded data to integer sequences for Embedding layer
    """
    return np.argmax(X_vectorized, axis=2)

def train_model_fold(model, X_train, y_train, X_dev, y_dev, fold):
    """
    Train the model for one fold with early stopping on development set
    """
    # Setup optimizer
    optimizer = Adam(lr=0.01)
    lr_metric = get_lr_metric(optimizer)
    
    # Compile model
    model.compile(
        loss="mse", 
        optimizer=optimizer, 
        metrics=[coeff_determination, lr_metric]
    )
    
    # Setup callbacks - same as MLP file
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-15,
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=300, 
        restore_best_weights=True
    )
    
    callbacks_list = [early_stopping, lr_scheduler]
    
    # Train model
    history = model.fit(
        x=X_train, 
        y=y_train,
        batch_size=32,
        epochs=2000,
        validation_data=(X_dev, y_dev),
        callbacks=callbacks_list,
        verbose=1
    )
    
    return history

def evaluate_fold(model, X_test, y_test):
    """
    Evaluate the model on test set and return all metrics like MLP file
    """
    y_pred = model.predict(X_test).flatten()
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    max_error = np.max(np.abs(y_test - y_pred))
    mre = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    return r2, mae, rmse, max_error, mre

def main():
    print("SMILES to Boiling Point Prediction using CNN with 10-fold Cross Validation")
    print("=" * 70)
    
    # Load data
    data_path = "matched_compounds_data_Vc.csv"
    data = load_data(data_path)
    
    # Extract SMILES and boiling points
    smiles_data = data['SMILES'].values
    boiling_points = data['Vc'].values*1000
    
    
    # Create character encoding
    charset, char_to_int, int_to_char, embed = create_charset_and_encoding(smiles_data)
    vocab_size = len(charset)
    
    X_vectorized = vectorize_smiles(smiles_data, char_to_int, embed, charset)

    # Prepare data for training (convert to integer sequences)
    X_sequences = prepare_data_for_training(X_vectorized)
    
    # Initialize cross-validation - same as MLP file
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    mae_scores = []
    rmse_scores = []
    max_error_scores = []
    mre_scores = []
    
    # Track best model - same as MLP file
    best_r2 = -float('inf')
    best_model = None
    
    # Cross-validation loop - same structure as MLP file
    for fold, (train_index, test_index) in enumerate(kf.split(X_sequences)):
        print(f"\n--- Fold {fold + 1}/10 ---")
        
        X_train_fold, X_temp = X_sequences[train_index], X_sequences[test_index]
        y_train_fold, y_temp = boiling_points[train_index], boiling_points[test_index]
        
        # Split temp into dev and test sets (50/50) - same as MLP file
        X_dev, X_test, y_dev, y_test, dev_indices, test_indices = train_test_split(
            X_temp, y_temp, test_index, test_size=0.5, random_state=fold
        )
        
        # Create model for this fold
        model = create_model(vocab_size, embed - 1)
        
        # Train model with early stopping on development set
        history = train_model_fold(model, X_train_fold, y_train_fold, X_dev, y_dev, fold)
        
        # Evaluate on test set
        r2, mae, rmse, max_error, mre = evaluate_fold(model, X_test, y_test)
        
        print(f"Fold {fold + 1} Results:")
        print(f"R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        print(f"Max Error: {max_error:.4f}, MRE: {mre:.2f}%")
        
        # Store metrics
        r2_scores.append(r2)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        max_error_scores.append(max_error)
        mre_scores.append(mre)
                
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
    
    print(f"\nBest model saved: R² = {best_r2:.4f}")
    
    # Calculate and display cross-validation results - same format as MLP file
    print(f"\n=== 10-Fold Cross Validation Results ===")
    print(f"Average R² Score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    print(f"Average MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Average Max Error: {np.mean(max_error_scores):.4f} ± {np.std(max_error_scores):.4f}")
    print(f"Average Mean Relative Error: {np.mean(mre_scores):.2f}% ± {np.std(mre_scores):.2f}%")
    
if __name__ == "__main__":
    main()