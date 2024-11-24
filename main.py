from src.textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_02_pdf_processing import PdfProcessingPipeline
from src.textSummarizer.pipeline.stage_03_text_preprocessing import TextPreProcessingPipeline
from src.textSummarizer.pipeline.stage_04_keyword_extraction import KeywordExtractionPipeline
from src.textSummarizer.pipeline.stage_05_text_processing import TextProcessingPipeline

from src.textSummarizer.logging import logger
from typing import List, Tuple
def run_data_ingestion_stage():
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<<")

    except Exception as e:
        logger.error(e)
        raise e




# Stage 2: Data Processing

def run_data_processing_stage():
    STAGE_NAME = "Data Processing Stage"
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        data_processing = PdfProcessingPipeline()
        processed_data = data_processing.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<<")
        return processed_data
    except Exception as e:
        logger.error(e)
        raise e

# Stage 3: Text Processing

def run_text_processing_stage(processed_data: List):
    STAGE_NAME = "Text Processing Stage"
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        text_processing = TextPreProcessingPipeline()
        text_processing.main(processed_data)
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<<")

    except Exception as e:
        logger.error(e)
        raise e

# Stage 4: Keyword Extraction

def run_keyword_extraction_stage():
    STAGE_NAME = "Keyword Extraction Stage"
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        keyword_extraction = KeywordExtractionPipeline()
        keyword_extraction.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<<")

    except Exception as e:
        logger.error(e)
        raise e
    
# Stage 5: Text Summanrization    
def run_text_processing_stage():
    STAGE_NAME = "Text Processing Stage"
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        text_processing = TextProcessingPipeline()
        chunks, chunked_embeddings = text_processing.main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<<<<")
        logger.info(f"Generated Summary: {chunks}")
        return chunks, chunked_embeddings

    except Exception as e:
        logger.error(e)
        raise e


if __name__ == "__main__":
    
    # Uncomment the stages to run them in sequence or as needed
    # run_data_ingestion_stage()
    # processed_data = run_data_processing_stage() #category, page_count, text 
    # run_text_processing_stage(processed_data)
    # run_keyword_extraction_stage()
    chunks, chunked_embeddings = run_text_processing_stage()