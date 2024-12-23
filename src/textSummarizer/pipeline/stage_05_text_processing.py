from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.text_processing import TextProcessor
from src.textSummarizer.logging import logger
import pandas as pd
import ast
# import faiss
import numpy as np

class TextProcessingPipeline:
    def __init__(self) -> None:
        """
        Initializes the Text Summarization Pipeline.
        Sets up the configuration manager to retrieve configurations needed for the summarization process.
        """
        self.config_manager = ConfigurationManager()

    def main(self,models):
        """
        Main method to run the text summarization pipeline.
        Executes the steps to process text, generate embeddings, retrieve relevant chunks, and summarize them.

        """
        try:
            # Retrieve configurations
            text_processing_config = self.config_manager.get_text_processing_config()

            # Instantiate the ModelLoader and TextProcessing components
            text_processor = TextProcessor(text_processing_config)

            logger.info("Starting the text summarization process.")

            # Load data
            df = pd.read_csv('keywords.csv')
            processed_text = df['processed_text'].values.tolist()[0]
            
            print("\n\n\n")
            print(processed_text,type(processed_text), len(processed_text))
            print("\n\n\n")
            # Load models
            
            # Process text and generate chunks
            # with Client(LocalCluster(n_workers=3, threads_per_worker=2,processes=True ,memory_limit='4GB')) as client:
            logger.info("Processing document for chunking.")
            chunks, _, _ = text_processor.semantic_chunking(phrases=processed_text,model= models[text_processing_config.sentence_model])
                # client.cancel(client.who_has())
            print("\n\n\nprinting chunks -----------------------------\n")
            print(chunks,type(chunks), len(chunks))
            print("\n\n\n")
            # Generate embeddings for chunks
            logger.info("Generating embeddings for document chunks.")
            chunked_embeddings = np.array([
                text_processor.get_embedding(
                    chunk,
                    models[text_processing_config.embedding_model_name],
                    models[text_processing_config.embedding_tokenizer]
                ) for chunk in chunks
            ])

            return chunks, chunked_embeddings            
        except Exception as e:
            logger.error(f"An error occurred in the Text Processing pipeline: {str(e)}")
            print(e.__traceback__)


