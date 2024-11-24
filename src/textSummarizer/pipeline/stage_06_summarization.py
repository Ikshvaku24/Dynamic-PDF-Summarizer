# pipeline/stage_06_rag_pipeline.py
from src.textSummarizer.config.configuration import ConfigurationManager
from src.textSummarizer.components.text_processing import TextProcessor
from src.textSummarizer.components.text_summarisation import TextSummarization
from src.textSummarizer.logging import logger
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from src.textSummarizer.components.query_generation import QueryGeneration
class TextSummarizationPipeline:
    def __init__(self):
        pass
    def main(self,
                query: str,
                chunks: List[str],
                chunked_embeddings: np.ndarray,
                models: Dict,
                device: str) -> List[str]:
        
        try:
            df = pd.read_csv('keywords.csv')
            keywords = df['keywords'].values.tolist()
            
            query_generator = QueryGeneration()
            query = query_generator.generate_domain_specific_query(chunks=chunks,keywords=keywords)
            # Initialize components
            summarizer = TextSummarization()
            #TODO start from here --
            # Create FAISS index
            faiss_index = summarizer.full_rag_pipeline(query=query, chunked_embeddings=chunked_embeddings, chunked_docs=chunks)
            
            # Get query embedding
            query_embedding = text_processor.get_embedding(
                query,
                models["longformer_model"],
                models["longformer_tokenizer"],
                device
            )

            # Retrieve similar chunks
            similar_chunks = retriever.retrieve_chunks(
                query_embedding,
                faiss_index,
                chunked_docs
            )

            # Rerank chunks
            reranked_chunks, _ = retriever.rerank_passages(
                query,
                similar_chunks,
                models["rerank_model"],
                models["rerank_tokenizer"],
                device
            )

            # Generate summaries
            final_summaries = [
                summarizer.generate_summary(
                    chunk,
                    models["distilbart_model"],
                    models["distilbart_tokenizer"]
                )
                for chunk in reranked_chunks
            ]

            return final_summaries

        except Exception as e:
            logger.error(f"Error in RAG pipeline execution: {str(e)}")
            return []