from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.pdf_processing import PdfProcessing
from src.textSummarizer.logging import logger
class PdfProcessingPipeline:
    def __init__(self) -> None:
        """
        Initializes the Data Processing Pipeline.
        Sets up the configuration manager to retrieve configurations needed for data processing.
        """
        self.config_manager = ConfigurationManager()
    
    def main(self):
        """
        Main method to run the data processing process.
        Processes multiple PDF files in the configured dataset directory.
        """
        try:
            data_processing_config = self.config_manager.get_data_processing_config()
            
            # Instantiate the DataProcessing component with the retrieved config
            data_processing = PdfProcessing(config=data_processing_config)
            
            logger.info("Starting the processing of multiple PDF files.")
            
            # Process multiple PDFs using the specified path
            x=data_processing.process_multiple_pdfs()
            print("len len\n",len(x),type(x))
            logger.info("Data processing completed successfully.")
            return x
        
        except Exception as e:
            logger.error(f"An error occurred in the Data Processing pipeline: {str(e)}")
