# pipeline/stage_06_rag_pipeline.py
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.text_summarisation import TextSummarization
from src.textSummarizer.logging import logger
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from src.textSummarizer.components.query_generation import QueryGeneration
from src.textSummarizer.components.text_processing import TextProcessor

class TextSummarizationPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        
    def main(self,
                chunks: List[str],
                chunked_embeddings: np.ndarray,
                models: Dict
            ) -> List[str]:
        
        try:
            df = pd.read_csv('keywords.csv')
            keywords = df['keywords'].values.tolist()[0].split(", ")
            query_generation_config= self.config_manager.get_query_generation_config()
            query_generator = QueryGeneration(query_generation_config)
            query = query_generator.generate_domain_specific_query(chunks=chunks,keywords=keywords, models=models)
            text_processing_config = self.config_manager.get_text_processing_config()

            query_embedding = TextProcessor(text_processing_config).get_embedding(
                query, 
                models[text_processing_config.embedding_model_name],
                models[text_processing_config.embedding_tokenizer]
            )
            
            # Initialize components
            text_summarization_config= self.config_manager.get_text_summarization_config()
            summarizer = TextSummarization(text_summarization_config)
           
            
            #TODO start from here --,standardise ,model cache, parallel processing
            # Create FAISS index
            return summarizer.full_rag_pipeline(query=query, chunked_embeddings=chunked_embeddings, chunked_docs=chunks,query_embedding=query_embedding, models=models)
           
        except Exception as e:
            logger.error(f"Error in RAG pipeline execution: {str(e)}")
            return []