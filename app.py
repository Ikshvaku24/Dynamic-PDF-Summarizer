import streamlit as st
from src.textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_02_pdf_processing import PdfProcessingPipeline
from src.textSummarizer.pipeline.stage_03_text_preprocessing import TextPreProcessingPipeline
from src.textSummarizer.pipeline.stage_04_keyword_extraction import KeywordExtractionPipeline
from src.textSummarizer.pipeline.stage_05_text_processing import TextProcessingPipeline
from src.textSummarizer.pipeline.stage_06_summarization import TextSummarizationPipeline
from src.textSummarizer.logging import logger

# Stage functions
def run_data_ingestion_stage():
    try:
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        st.success("Data Ingestion Stage completed successfully!")
    except Exception as e:
        st.error(f"Error in Data Ingestion Stage: {e}")

def run_data_processing_stage():
    try:
        data_processing = PdfProcessingPipeline()
        processed_data = data_processing.main()
        st.success("Data Processing Stage completed successfully!")
        return processed_data
    except Exception as e:
        st.error(f"Error in Data Processing Stage: {e}")
        return None

def run_text_preprocessing_stage(processed_data):
    try:
        text_processing = TextPreProcessingPipeline()
        text_processing.main(processed_data)
        st.success("Text Preprocessing Stage completed successfully!")
    except Exception as e:
        st.error(f"Error in Text Processing Stage: {e}")

def run_keyword_extraction_stage():
    try:
        keyword_extraction = KeywordExtractionPipeline()
        keyword_extraction.main()
        st.success("Keyword Extraction Stage completed successfully!")
    except Exception as e:
        st.error(f"Error in Keyword Extraction Stage: {e}")

def run_text_processing_pipeline():
    try:
        text_processing = TextProcessingPipeline()
        chunks, chunked_embeddings, models = text_processing.main()
        st.success("Text Processing Pipeline completed successfully!")
        return chunks, chunked_embeddings, models
    except Exception as e:
        st.error(f"Error in Text Processing Pipeline: {e}")
        return None, None, None

def run_text_summarization_stage(chunks, chunked_embeddings, models):
    try:
        text_summarization = TextSummarizationPipeline()
        final_summaries = text_summarization.main(chunks=chunks, chunked_embeddings=chunked_embeddings, models=models)
        st.success("Text Summarization Stage completed successfully!")
        st.text_area("Generated Summary", final_summaries, height=200)
    except Exception as e:
        st.error(f"Error in Text Summarization Stage: {e}")

# Streamlit UI
def main():
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

    # Stage 3: Text Processing
    if st.button("Run Text Processing Stage"):
        if "processed_data" in st.session_state:
            run_text_preprocessing_stage(st.session_state["processed_data"])
        else:
            st.warning("Please run the Data Processing Stage first.")

    # Stage 4: Keyword Extraction
    if st.button("Run Keyword Extraction Stage"):
        run_keyword_extraction_stage()

    # Stage 5: Text Processing Pipeline
    if st.button("Run Text Processing Pipeline"):
        chunks, chunked_embeddings, models = run_text_processing_pipeline()
        if chunks:
            st.session_state["chunks"] = chunks
            st.session_state["chunked_embeddings"] = chunked_embeddings
            st.session_state["models"] = models

    # Stage 6: Text Summarization
    if st.button("Run Text Summarization Stage"):
        if "chunks" in st.session_state and "chunked_embeddings" in st.session_state and "models" in st.session_state:
            run_text_summarization_stage(
                chunks=st.session_state["chunks"], 
                chunked_embeddings=st.session_state["chunked_embeddings"], 
                models=st.session_state["models"]
            )
        else:
            st.warning("Please run the Text Processing Pipeline first.")

if __name__ == "__main__":
    main()
