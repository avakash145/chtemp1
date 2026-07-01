import pandas as pd
import logging
import os
import nltk
from nltk.stem import PorterStemmer
import string
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
ps=PorterStemmer()
logs_dir='../logs'
os.makedirs(logs_dir,exist_ok=True)
logger=logging.getLogger('transforms')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

logfilepath=os.path.join(logs_dir,'transform.log')
filehandler=logging.FileHandler(logfilepath)
filehandler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
filehandler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(filehandler)


def transform_text(text:string)->string:
    """transforms the input by converting it to lower case, removing stopwords and punctuation and stemming the words"""
    text=text.lower()
    text=nltk.word_tokenize(text)
    text=[i for i in text if i.isalnum()]
    text=[i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text=[ps.stem(i) for i in text]
    return " ".join(text)


def main(text_column='text',target_column='target'):
    """transforms the input text column and returns the transformed dataframe"""
    try:
        # fetch the data from the dataingested dir 
        train_path='../data/raw/train.csv'
        test_path='../data/raw/test.csv'
        td=pd.read_csv(train_path,encoding='latin')
        ted=pd.read_csv(test_path,encoding='latin')
        logger.info("data loaded from %s and %s",train_path,test_path)
        
        # transform the text column in both train and test data
        td['text']=td.text.apply(transform_text)
        ted['text']=ted.text.apply(transform_text)
        datapath='../data/transformed'
        os.makedirs(datapath,exist_ok=True)
        td.to_csv(os.path.join(datapath, 'train_transformed.csv'), index=False)
        ted.to_csv(os.path.join(datapath, 'test_transformed.csv'), index=False)
        logger.info("text column transformed in both train and test data and saved to %s",datapath)
    except FileNotFoundError as e:
        logger.error("file not found: %s", str(e))
        raise
    except pd.errors.ParserError as e:
        logger.error("failed to parse the csv file: %s", str(e))
        raise
    except pd.errors.EmptyDataError as e:
        logger.error("no data in the csv file: %s", str(e))
        raise
    except Exception as e:
        logger.error("error occurred while transforming text column: %s", str(e))
        raise

if __name__=="__main__":
    main()

    






