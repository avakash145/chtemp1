import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier 
log_dir="logs"


logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)
os.makedirs(log_dir, exist_ok=True)

file_handler = logging.FileHandler(os.path.join(log_dir, "model_training.log"))
file_handler.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"No data: {file_path} is empty")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
    
def model_train(X_train:np.ndarray, y_train:np.ndarray, params:dict)->RandomForestClassifier:
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must be the same")
        logger.debug(f"Initializing RandomForestClassifier with parameters: {params}")
        model = RandomForestClassifier(**params)
        logger.debug(f"Starting model training with {X_train.shape[0]} samples")
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully")
        return model
    except ValueError as ve:
        logger.error(f"ValueError during model training: {ve}")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

def save_model(model:RandomForestClassifier, file_path:str):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        logger.debug(f"Model saved successfully to {file_path}")
    except FileNotFoundError:
        logger.error(f"Directory not found for saving model: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}")
        raise
    
def main():
    try:
        params={'n_estimators': 26,'random_state': 2}
        train_data_path = "data/feature_engineered/train_feature_engineered.csv"
        model_save_path = "models/model.pkl"
        train_data = load_data(train_data_path)
        x_train =train_data.drop(columns=['target']).values
        y_train = train_data['target'].values
        model = model_train(x_train, y_train, params)
        save_model(model, model_save_path)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
if __name__ == "__main__":
    main()