import streamlit as st
from src.textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_02_pdf_processing import PdfProcessingPipeline
from src.textSummarizer.pipeline.stage_03_text_preprocessing import TextPreProcessingPipeline
from src.textSummarizer.pipeline.stage_04_keyword_extraction import KeywordExtractionPipeline
from src.textSummarizer.pipeline.stage_05_text_processing import TextProcessingPipeline
from src.textSummarizer.pipeline.stage_06_summarization import TextSummarizationPipeline
from src.textSummarizer.components.model_loader import ModelLoader
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.logging import logger

# Stage functions
def run_data_ingestion_stage():
    try:
        st.info("Running Data Ingestion Stage...")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        st.success("Data Ingestion Stage completed successfully!")
    except Exception as e:
        st.error(f"Error in Data Ingestion Stage: {e}")

def run_data_processing_stage():
    try:
        st.info("Running Data Processing Stage...")
        data_processing = PdfProcessingPipeline()
        processed_data = data_processing.main()
        st.success("Data Processing Stage completed successfully!")
        return processed_data
    except Exception as e:
        st.error(f"Error in Data Processing Stage: {e}")
        return None

def run_text_preprocessing_stage(processed_data):
    try:
        st.info("Running Text Preprocessing Stage...")
        text_processing = TextPreProcessingPipeline()
        text_processing.main(processed_data)
        st.success("Text Preprocessing Stage completed successfully!")
    except Exception as e:
        st.error(f"Error in Text Preprocessing Stage: {e}")

def run_keyword_extraction_stage():
    try:
        st.info("Running Keyword Extraction Stage...")
        keyword_extraction = KeywordExtractionPipeline()
        keyword_extraction.main()
        st.success("Keyword Extraction Stage completed successfully!")
    except Exception as e:
        st.error(f"Error in Keyword Extraction Stage: {e}")

def run_text_processing_stage(models):
    try:
        print("hihi")
        st.info("Running Text Processing Stage...")
        text_processing = TextProcessingPipeline()
        chunks, chunked_embeddings = text_processing.main(models)
        st.success("Text Processing Stage completed successfully!")
        return chunks, chunked_embeddings
    except Exception as e:
        st.error(f"Error in Text Processing Stage: {e}")
        return None, None

def run_text_summarization_stage(chunks, chunked_embeddings, models):
    try:
        st.info("Running Text Summarization Stage...")
        text_summarization = TextSummarizationPipeline()
        final_summaries = text_summarization.main(chunks=chunks, chunked_embeddings=chunked_embeddings, models=models)
        st.success("Text Summarization Stage completed successfully!")
        st.text_area("Generated Summary", final_summaries, height=200)
    except Exception as e:
        st.error(f"Error in Text Summarization Stage: {e}")

# Streamlit UI
def main():
    # Initialize configuration manager and model loader
    config_manager = ConfigurationManager()
    model_config = config_manager.get_model_config()
    model_loader = ModelLoader(model_config)
    models = model_loader.load_models()

    # Title and description
    st.title("Text Summarizer Pipeline")
    st.write("Run each stage of the pipeline sequentially using the buttons below:")

    # Stage 1: Data Ingestion
    if st.button("Run Data Ingestion Stage"):
        run_data_ingestion_stage()

    # Stage 2: Data Processing
    if st.button("Run Data Processing Stage"):
        processed_data = run_data_processing_stage()
        if processed_data:
            st.session_state["processed_data"] = processed_data

    # Stage 3: Text Preprocessing
    if st.button("Run Text Preprocessing Stage"):
        if "processed_data" in st.session_state:
            run_text_preprocessing_stage(st.session_state["processed_data"])
        else:
            st.warning("Please run the Data Processing Stage first.")

    # Stage 4: Keyword Extraction
    if st.button("Run Keyword Extraction Stage"):
        run_keyword_extraction_stage()

    # Stage 5: Text Processing
    if st.button("Run Text Processing Stage"):
        chunks, chunked_embeddings = run_text_processing_stage(models)
        if chunks:
            st.session_state["chunks"] = chunks
            st.session_state["chunked_embeddings"] = chunked_embeddings

    # Stage 6: Text Summarization
    if st.button("Run Text Summarization Stage"):
        if "chunks" in st.session_state and "chunked_embeddings" in st.session_state:
            run_text_summarization_stage(
                chunks=st.session_state["chunks"], 
                chunked_embeddings=st.session_state["chunked_embeddings"],
                models=models 
            )
        else:
            st.warning("Please run the Text Processing Stage first.")

if __name__ == "__main__":
    main()
