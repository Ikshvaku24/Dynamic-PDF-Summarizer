from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.text_preprocessing import TextPreProcessing
from src.textSummarizer.logging import logger
from typing import List
class TextPreProcessingPipeline:
    def __init__(self) -> None:
        """
        Initializes the Text Processing Pipeline.
        Sets up the configuration manager to retrieve configurations needed for text preprocessing.
        """
        self.config_manager = ConfigurationManager()

    def main(self, processed_data: List):
        """
        Main method to run the text processing pipeline.
        Preprocesses the text data from the configured dataset.
        """
        try:
            text_preprocessing_config = self.config_manager.get_text_preprocessing_config()

            # Instantiate the TextProcessing component with the retrieved config
            text_preprocessing = TextPreProcessing(config=text_preprocessing_config)

            logger.info("Starting the text preprocessing.")
            
            # Process the text using the text processing component
            text_preprocessing.process_text_file(processed_data)

            logger.info("Text processing completed successfully.")
        
        except Exception as e:
            logger.error(f"An error occurred in the Text Processing pipeline: {str(e)}")
