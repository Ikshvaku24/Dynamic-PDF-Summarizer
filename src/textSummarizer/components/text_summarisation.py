import torch
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from src.textSummarizer.logging import logger
from src.textSummarizer.config.configuration import TextSummarizationConfig
class TextSummarization:
    def __init__(self, config: TextSummarizationConfig):
        self.config = config
        

    def create_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        
        try:
            embedding_dim = embeddings.shape[1]
            index = faiss.IndexFlatL2(embedding_dim)
            index.add(embeddings)
            return index
        except Exception as e:
            logger.error(f"Error in creating FAISS index: {str(e)}")
            return None
        
    def retrieve_chunks(self, 
                       query_embedding: np.ndarray, 
                       faiss_index: faiss.IndexFlat, 
                       chunked_docs: List[str], 
                       ) -> List[str]:
        """Retrieve relevant chunks using FAISS."""
        try:
            top_k = self.config.chunking_top_k
            query_embedding = np.array([query_embedding])
            faiss.normalize_L2(query_embedding)
            distances, indices = faiss_index.search(query_embedding, top_k)
            similar_chunks = [chunked_docs[i] for i in indices[0]]
            return similar_chunks
        except Exception as e:
            logger.error(f"Error in chunk retrieval: {str(e)}")
            return []

    def rerank_passages(self, 
                       query: str, 
                       passages: List[str], 
                       rerank_model, 
                       device: str, 
                       rerank_tokenizer 
                       ) -> Tuple[List[str], List[float]]:
        """Rerank passages using cross-encoder model."""
        try:
            top_k = self.config.reranking.top_k
            scores = []
            for passage in passages:
                inputs = rerank_tokenizer(
                    query, 
                    passage, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}

                with torch.no_grad():
                    outputs = rerank_model(**inputs)
                
                score = outputs.logits.squeeze().item()
                scores.append(score)

            # Sort passages by score
            ranked_pairs = sorted(zip(scores, passages), 
                                key=lambda x: x[0], 
                                reverse=True)
            ranked_passages = [passage for _, passage in ranked_pairs[:top_k]]
            ranked_scores = [score for score, _ in ranked_pairs[:top_k]]

            return ranked_passages, ranked_scores
        except Exception as e:
            logger.error(f"Error in passage reranking: {str(e)}")
            return [], []

    def generate_summary(self, 
                        text: str, 
                        model, 
                        tokenizer 
                        ) -> str:
        """Generate summary using the specified model."""
        try:
            max_length = self.config.summarization.max_length
            min_length = self.config.summarization.min_length
            inputs = tokenizer(text, 
                             return_tensors='pt', 
                             max_length=1024, 
                             truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            summary_ids = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                min_length=min(min_length,max_length//2),
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
            
            return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error in summary generation: {str(e)}")
            return ""

    def full_rag_pipeline(self, 
                         query: str, 
                         chunked_embeddings: np.ndarray, 
                         chunked_docs: List[str],
                         query_embedding: np.ndarray, 
                         models: Dict 
                         ) -> List[str]:
        """Execute the full RAG pipeline."""
        try:
            device = self.config.device
            
            faiss_index = self.create_faiss_index(chunked_embeddings)
            

            # Retrieve similar chunks
            similar_chunks = self.retrieve_chunks(
                query_embedding, 
                faiss_index, 
                chunked_docs 
            )

            # Rerank chunks
            reranked_chunks, _ = self.rerank_passages(
                query=query, 
                passages=similar_chunks, 
                rerank_model=models[self.config.reranking.rerank_model], 
                rerank_tokenizer=models[self.config.reranking.rerank_tokenizer],
                device=device
            )

            # Generate summaries
            final_summaries = [
                self.generate_summary(
                    chunk, 
                    models[self.config.summarization.summarization_model], 
                    models[self.config.summarization.summarization_tokenizer]
                ) 
                for chunk in reranked_chunks
            ]

            return final_summaries
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return []