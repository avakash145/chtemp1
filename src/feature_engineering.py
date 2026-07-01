import pandas as pd
import os 
import logging 
from sklearn.feature_extraction.text import TfidfVectorizer

logs_dir='../logs'
os.makedirs(logs_dir, exist_ok=True)

logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(logs_dir,'feature_engineering.log')
filehandler=logging.FileHandler(log_file_path)
filehandler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(filehandler)


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a csv file."""
    try:
        df = pd.read_csv(data_path, encoding='latin')
        df.fillna('', inplace=True)  # Fill NaN values with empty strings
        logger.debug('Data loaded from %s', data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", str(e))
        raise
    except FileNotFoundError as e:
        logger.error("File not found: %s", str(e))
        raise
    except Exception as e:
        logger.error("Unexpected error while loading the data: %s", str(e))
        raise
    
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features:int) ->tuple:
    """Apply TF-IDF vectorization to the text column of train and test data."""
    try:
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        xt=train_data.text.values
        xte=test_data.text.values
        yt=train_data.target.values
        yte=test_data.target.values
        
        xtbow=tfidf_vectorizer.fit_transform(xt)
        xtebow=tfidf_vectorizer.transform(xte)
        train_df=pd.DataFrame(xtbow.toarray())
        train_df['target']=yt
        test_df=pd.DataFrame(xtebow.toarray())
        test_df['target']=yte
        logger.debug("TF-IDF vectorization applied to %s column", 'text')
        return train_df,test_df
    except KeyError as e:
        logger.error("Missing column in the dataframe: %s", str(e))
        raise
    except Exception as e:
        logger.error("Error occurred while applying TF-IDF vectorization: %s", str(e))
        raise

def save_data(df:pd.DataFrame,file_path:str)->None:
    """Save the dataframe to a csv file."""
    try:
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug("Data saved to %s", file_path)
    except Exception as e:
        logger.error("Error occurred while saving the data: %s", str(e))
        raise

def main():
    try:
        max_features=80
        train_path='../data/transformed/train_transformed.csv'
        test_path='../data/transformed/test_transformed.csv'
        train_data=load_data(train_path)
        test_data=load_data(test_path)
        train_df,test_df=apply_tfidf(train_data,test_data,max_features)
        save_data(train_df,'../data/feature_engineered/train_feature_engineered.csv')
        save_data(test_df,'../data/feature_engineered/test_feature_engineered.csv')
    except Exception as e:
        logger.error("Failed to complete the feature engineering process: %s", str(e))
        print(f"Error: {e}")

if __name__=='__main__':
    main()