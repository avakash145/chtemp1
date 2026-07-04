import os 
import numpy as np 
import pandas as pd 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score,roc_curve
import logging
import pickle
import yaml
import dvclive
from dvclive import Live
import json


logs_dir="logs"
logger=logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

os.makedirs(logs_dir, exist_ok=True)

filehandler=logging.FileHandler(os.path.join(logs_dir,"model_evaluation.log"))
filehandler.setLevel('DEBUG')
consolehandler=logging.StreamHandler()
consolehandler.setLevel('DEBUG')
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
filehandler.setFormatter(formatter)
consolehandler.setFormatter(formatter)
logger.addHandler(filehandler)
logger.addHandler(consolehandler)

def load_params(file_path:str)->dict:
    try:
        with open(file_path,'r') as f:
            params=yaml.safe_load(f)
            logger.debug(f"Parameters loaded successfully from {file_path}")
            return params
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters from {file_path}: {e}")
        raise




def load_model(file_path:str)->pd.DataFrame:
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Model loaded successfully from {file_path}")
        return model
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {file_path}: {e}")
        raise
def load_data(file_path:str)->pd.DataFrame:
    try:
        data=pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
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
    
    
def evaluate_model(model, X_test:np.ndarray, y_test:np.ndarray)->dict:
    try:
        logger.debug(f"Starting model evaluation on {X_test.shape[0]} samples")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        logger.info(f"Model evaluation completed successfully with accuracy: {accuracy}, ROC AUC: {roc_auc}")
        
        return {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report
        }
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
    

def main():
    try:
        model_path = "models/model.pkl"
        params_path = "params.yaml"
        params=load_params(params_path)
        test_data_path = "data/feature_engineered/test_feature_engineered.csv"
        model = load_model(model_path)
        test_data = load_data(test_data_path)
        X_test = test_data.drop("target", axis=1).values
        y_test = test_data["target"].values
        
        evaluation_results = evaluate_model(model, X_test, y_test)
        
        results_path = "results/evaluation_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        logger.debug(f"Evaluation results saved successfully to {results_path}")
        with Live(save_dvc_exp=True) as live:
            live.log_metric("classification_report",evaluation_results['classification_report'])
            live.log_metric("accuracy",evaluation_results['accuracy'])
            live.log_metric("roc_auc",evaluation_results['roc_auc'])
            
            live.log_params(params)
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise
if __name__ == "__main__":
    main()
