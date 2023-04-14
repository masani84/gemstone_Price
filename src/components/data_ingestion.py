import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation

## Initilize the data ingetion configeration        ## artifacts: folder name
@dataclass
class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts', 'train.csv')
    test_data_path:str=os.path.join('artifacts', 'test.csv')
    raw_data_path:str=os.path.join('artifacts', 'raw.csv')

## Create a class for Data Ingestion
class Dataingestion:
    def __init__(self):
        self.ingestion_config=DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method Starts')
        
        try:
            #df = pd.read_csv('notebooks/data/gemstone.csv')
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info("Dataset read from notebooks")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info('Train test Split')  
            train_set, test_set=train_test_split(df, test_size=0.3, random_state=30)
            
            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True)
            
            logging.info('Data Ingestion is completed')     

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )   

        
        except Exception as e:
            logging.info('Exception occured at data ingestion stage')
            raise CustomException(e,sys)

'''      
        ## run data ingestion 

if __name__=='__main__':
    obj=Dataingestion()
    train_data, test_data = obj.initiate_data_ingestion()

'''

        ## run data tranformation 

if __name__=='__main__':
    obj=Dataingestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.initaite_data_transformation(train_data,test_data)








