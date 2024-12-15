from dataclasses import dataclass
from pathlib import Path
from typing import Dict

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    local_data_file: Path
    failed_data_file: Path
    urls: Dict[str,str]

@dataclass(frozen=True)
class PdfProcessingConfig:
    pdf_folder_path: Path
    batch_size: int
    max_retries: int
    wait_time: int
    

@dataclass(frozen=True)
class KeywordExtractionConfig:
    remove_newline: bool
    stop_words: bool
    language: str
    max_ngram_size: int
    deduplication_threshold: float
    deduplication_algo: str
    window_size: int

@dataclass(frozen=True)
class RapidfuzzConfig:
    threshold: int

@dataclass(frozen=True)
class TextPreProcessingConfig:
    remove_html: bool
    remove_urls: bool
    remove_emails: bool
    handle_accents: bool
    remove_unicode_chars: bool
    remove_punctuations: bool
    remove_digits: bool
    remove_extra_spaces: bool
    correct_spelling: bool
    expand_contractions: bool
    
@dataclass
class ModelConfig:
    longformer_model_name: str
    distilbart_model_name: str
    sentence_model_name: str
    rerank_model_name: str

@dataclass
class TextProcessingConfig:
    batch_size: int
    dynamic_percentile: int
    max_tokens: int
    embedding_model_name: str
    embedding_tokenizer: str
    sentence_model: str
    tokenizer_max_length: int
    device: str
    padding: str
    truncation: bool

@dataclass
class QueryGenerationConfig:
    most_common_words_count: int  
    num_topics: int 
    
@dataclass
class RerankingConfig:
    top_k: int
    rerank_model: str
    rerank_tokenizer: str

@dataclass
class SummarizationConfig:
    summarization_model: str
    summarization_tokenizer: str
    max_length: int
    min_length: int
    tokenizer_max_length: int

@dataclass
class TextSummarizationConfig:
    device: str
    chunking_top_k: int
    summarization: SummarizationConfig
    reranking: RerankingConfig 

         