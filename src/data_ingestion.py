import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
lblencoder=LabelEncoder()
logs_dir='logs'
os.makedirs(logs_dir, exist_ok=True)

logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(logs_dir,'data_ingestion.log')
filehandler=logging.FileHandler(log_file_path)
filehandler.setLevel('DEBUG')

formatter= logging.Formatter('%(asctime)s - %(name)s-%(levelname)s-%(message)s')
console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(filehandler)





def load_data(data_url: str)-> pd.DataFrame:
    '''Load data from a csv file. '''
    try:
        df=pd.read_csv(data_url,encoding='latin')
        logger.debug('Data loaded from %s',data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error(f'faile to parse the csv file {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected error while loading the data {e}')
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    '''preprocess the data'''
    try:
        df=df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'])
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        df['target']=lblencoder.fit_transform(df.target)
        df=df.drop_duplicates(keep='first')
        logger.debug('Data preprocessing complete')
        return df
    except KeyError as e:
        logger.error(f'Missing column in the dataframe {e}')
        raise
    except Exception as e:
        logger.error(f'Unexpected Error raised: {e}')
        raise
    
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    try: 
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)    
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)    
        logger.debug(f"train and test data saved to dir: {raw_data_path} ")
    except Exception as e:
        logger.error(f'Unexpected errror raised : {e}')
        raise

def main():
    try:
        test_size=0.2
        data_path='experiments/spam.csv'
        # df=load_data(data_url=data_path)  where data path is the url to the data 
        df=load_data(data_path)
        df=preprocess_data(df)
        train_data,test_data=train_test_split(df,test_size=0.2,random_state=2)
        save_data(train_data,test_data,"data")
    except Exception as e:
        logger.error(f"Failed to complete the data ingestion process : {e}")
        print(f"Error {e}")

if __name__=='__main__':
    main()



