from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.data_ingestion import DataIngestion
from src.textSummarizer.logging import logger
import traceback
class DataIngestionTrainingPipeline:
    def __init__(self) -> None:
        """
        Initializes the Data Ingestion Training Pipeline.
        Sets up the configuration manager to retrieve configurations needed for data ingestion.
        """
        self.config_manager = ConfigurationManager()
    
    def main(self):
        """
        Main method to run the data ingestion process.
        Downloads multiple files in parallel as configured.
        """
        try:
            # Fetch the data ingestion configuration from the configuration manager
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            
            # Instantiate the DataIngestion component with the retrieved config
            data_ingestion = DataIngestion(config=data_ingestion_config)
            
            logger.info("Starting the download process for multiple files.")
            
            # Download multiple files in parallel using the max_workers parameter
            data_ingestion.download_multiple_files()
            
            logger.info("Data ingestion completed successfully.")
        
        except Exception as e:
            logger.error(f"An error occurred in the Data Ingestion pipeline: {str(e)}")
            logger.error(traceback.format_exc())