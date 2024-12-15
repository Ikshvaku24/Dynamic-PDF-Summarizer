from src.textSummarizer.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, urls
from src.textSummarizer.utils.common import read_yaml, create_directories
from src.textSummarizer.entity import (
    DataIngestionConfig,
    PdfProcessingConfig,
    TextPreProcessingConfig,
    KeywordExtractionConfig,
    RapidfuzzConfig,
    ModelConfig,
    TextProcessingConfig,
    QueryGenerationConfig,
    TextSummarizationConfig,
    RerankingConfig,
    SummarizationConfig   
)
from typing import Dict

from pathlib import Path

class ConfigurationManager:
    def __init__(
            self,
            urls: Dict[str,str] = urls,
            config_filepath: Path = CONFIG_FILE_PATH,
            params_filepath: Path = PARAMS_FILE_PATH
        ) -> None:
        self.urls = urls
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([Path(self.config.artifacts_root)])  # Ensure the root directory exists
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir), Path(config.local_data_file), Path(config.failed_data_file)])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            local_data_file=Path(config.local_data_file),
            failed_data_file=Path(config.failed_data_file),
            urls=self.urls
        )
        
        return data_ingestion_config
    
    
    def get_data_processing_config(self) -> PdfProcessingConfig:
        config = self.config.data_processing
        params = self.params.data_processing
        data_processing_config = PdfProcessingConfig(
            pdf_folder_path = config.pdf_folder_path,
            batch_size = params.batch_size,
            max_retries = params.max_retries,
            wait_time = params.wait_time
        )
        return data_processing_config
        
    def get_text_preprocessing_config(self) -> TextPreProcessingConfig:
        params = self.params.text_preprocessing
        text_processing_config = TextPreProcessingConfig(
            remove_html=params.remove_html,
            remove_urls=params.remove_urls,
            remove_emails=params.remove_emails,
            handle_accents=params.handle_accents,
            remove_unicode_chars=params.remove_unicode_chars,
            remove_punctuations=params.remove_punctuations,
            remove_digits=params.remove_digits,
            remove_extra_spaces=params.remove_extra_spaces,
            correct_spelling=params.correct_spelling,
            expand_contractions=params.expand_contractions
        )
        return text_processing_config

    def get_keyword_extraction_config(self) -> KeywordExtractionConfig:
        params = self.params.keyword_extraction
        return KeywordExtractionConfig(
            remove_newline=params.remove_newline,
            stop_words=params.stop_words,
            language=params.language,
            max_ngram_size=params.max_ngram_size,
            deduplication_threshold=params.deduplication_threshold,
            deduplication_algo=params.deduplication_algo,
            window_size=params.window_size
        )

    def get_rapidfuzz_config(self) -> RapidfuzzConfig:
        params = self.params.rapidfuzz
        return RapidfuzzConfig(
            threshold=params.threshold
        )
    
    def get_model_config(self) -> ModelConfig:
        config = self.config.model_config
        return ModelConfig(
            longformer_model_name=config.longformer_model_name,
            distilbart_model_name=config.distilbart_model_name,
            sentence_model_name=config.sentence_model_name,
            rerank_model_name=config.rerank_model_name
        )
    def get_text_processing_config(self) -> TextProcessingConfig:
        params = self.params.text_processing
        return TextProcessingConfig(
            batch_size=params.batch_size,                       # Using the provided batch size
            dynamic_percentile=params.dynamic_percentile,       # Dynamic percentile from config
            max_tokens=params.max_tokens,                       # Max tokens as defined in the config
            embedding_model_name=params.embedding_model_name,   # Embedding model name as defined
            embedding_tokenizer=params.embedding_tokenizer,     # Embedding tokenizer as defined
            sentence_model=params.sentence_model,     
            tokenizer_max_length=params.tokenizer_max_length,   # Tokenizer max length as per config
            device=params.device,                               # Device for model processing
            padding=params.padding,                             # Padding strategy from config
            truncation=params.truncation                        # Truncation flag from config
        )
    def get_query_generation_config(self) -> QueryGenerationConfig:
        params = self.params.query_generation  # Assuming params.query_generation exists
        return QueryGenerationConfig(
            most_common_words_count=params.most_common_words_count,  # Top N most common words
            num_topics=params.num_topics                    # Number of topics for LDA
        )

    def get_text_summarization_config(self) -> TextSummarizationConfig:
        params = self.params.text_summarization  # Assuming params.text_summarization exists
        # Extracting the summarization and reranking configurations
        summarization_params = params.summarization
        reranking_params = params.reranking
        
        # Creating the SummarizationConfig and RerankingConfig objects
        summarization_config = SummarizationConfig(
            summarization_model=summarization_params.summarization_model,
            summarization_tokenizer=summarization_params.summarization_tokenizer,
            max_length=summarization_params.max_length,
            min_length=summarization_params.min_length,
            tokenizer_max_length=summarization_params.tokenizer_max_length
        )
        
        reranking_config = RerankingConfig(
            top_k=reranking_params.top_k,
            rerank_model=reranking_params.rerank_model,
            rerank_tokenizer=reranking_params.rerank_tokenizer
        )
        
        # Returning the final TextSummarizationConfig with both summarization and reranking configurations
        return TextSummarizationConfig(
            device = params.device,
            chunking_top_k=params.chunking_top_k,
            summarization=summarization_config,
            reranking=reranking_config
        )



