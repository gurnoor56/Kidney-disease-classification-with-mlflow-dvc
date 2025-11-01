import os
import zipfile
import gdown
from KidneyClassification import logger
from KidneyClassification.utils.common import get_size
from KidneyClassification.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        fetch data from url
        '''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading file from : {dataset_url} into file : {zip_download_dir}")

            file_id = dataset_url.split('/')[-2]
            prefix = "https://drive.google.com/uc?export=download&id="
            gdown.download(prefix + file_id, str(zip_download_dir),quiet=False)

            logger.info(f"downloaded data from {dataset_url} into file : {zip_download_dir}")

        except Exception as e:
            raise e
        

    def extract_zip_file(self):
        """
        zip file path:str
        extracts the zip file  into the directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

   