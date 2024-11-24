from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.keyword_extraction import KeywordExtraction
from src.textSummarizer.logging import logger

class KeywordExtractionPipeline:
    def __init__(self) -> None:
        """
        Initializes the Keyword Extraction Pipeline.
        Sets up the configuration manager to retrieve configurations needed for keyword extraction.
        """
        self.config_manager = ConfigurationManager()

    def main(self):
        """
        Main method to run the keyword extraction process.
        Extracts keywords from the preprocessed text data.
        """
        try:
            keyword_extraction_config = self.config_manager.get_keyword_extraction_config()
            rapidfuzz_config = self.config_manager.get_rapidfuzz_config()

            # Instantiate the KeywordExtraction component with the retrieved config
            keyword_extraction = KeywordExtraction(keyword_config=keyword_extraction_config, rapidfuzz_config=rapidfuzz_config)

            logger.info("Starting the keyword extraction process.")

            # Extract keywords using the keyword extraction component
            extracted_keywords = keyword_extraction.extract_keywords_from_file()

            logger.info("Keyword extraction completed successfully.")
        
        except Exception as e:
            logger.error(f"An error occurred in the Keyword Extraction pipeline: {str(e)}")