# pipeline/stage_06_rag_pipeline.py
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.text_summarisation import TextSummarization
from src.textSummarizer.logging import logger
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
import torch
from src.textSummarizer.components.query_generation import QueryGeneration
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
            keywords = df['keywords'].values.tolist()
            
            query_generator = QueryGeneration()
            query = query_generator.generate_domain_specific_query(chunks=chunks,keywords=keywords)
            # Initialize components
            summarizer = TextSummarization()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
           
            
            #TODO start from here --,standardise ,model cache, parallel processing
            # Create FAISS index
            return summarizer.full_rag_pipeline(query=query, chunked_embeddings=chunked_embeddings, chunked_docs=chunks, models=models, device=device)
           
        except Exception as e:
            logger.error(f"Error in RAG pipeline execution: {str(e)}")
            return []