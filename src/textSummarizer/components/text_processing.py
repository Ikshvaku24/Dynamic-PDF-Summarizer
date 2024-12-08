import numpy as np
import torch
from typing import List, Tuple, Dict, Any
import logging
from src.textSummarizer.logging import logger
from src.textSummarizer.components.model_loader import ModelLoader
class TextProcessor:
    def __init__(self):
        pass
        # self.config = config

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(text.split())

    def calculate_similarity_score(self, pair: Tuple[np.ndarray, np.ndarray]) -> float:
        """Calculate cosine similarity between two embeddings."""
        emb1, emb2 = pair
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

    def find_dynamic_threshold(self, similarity_scores: List[float], percentile: int = 20) -> float:
        """Find dynamic threshold based on similarity score distribution."""
        if not similarity_scores:
            return 0.0
        return float(np.percentile(similarity_scores, percentile))

    def process_batch(self, phrases: List[str], model) -> np.ndarray:
        """Process a batch of phrases to get embeddings."""
        if not phrases:
            return np.array([])
        try:
            return model.encode(phrases)
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            return np.array([])

    def semantic_chunking(self, 
                         phrases: str, 
                         model,
                         batch_size: int = 500, 
                         dynamic_percentile: int = 20,
                         max_tokens: int = 4096) -> Tuple[List[str], List[float], float]:
        """
        Perform semantic chunking on the input phrases.
        """
        if isinstance(phrases, str):
            logger.warning("Input is a single string, splitting into sentences.")
            phrases = phrases.splitlines()
        if not phrases:
            logger.warning("No phrases found!")
            return [], [], 0.0

        logger.info(f"Processing {len(phrases)} phrases")

        try:
            # Process batches
            print("phrases-",len(phrases),type(phrases))
            #here batches are created. batch contains lines it is a list not str
            batches = [phrases[i:i+batch_size] for i in range(0, len(phrases), batch_size)]
            encoded_phrases = []
            for batch in batches:
                encoded_phrases.append(self.process_batch(batch, model))
            
            # Flatten the list of encoded phrases (assuming process_batch returns a 2D array)
            encoded_phrases = np.vstack([emb for emb in encoded_phrases if len(emb) > 0])

            if len(encoded_phrases) <= 1:
                logger.warning("Insufficient embeddings generated")
                return [], [], 0.0

            # Calculate similarity scores
            phrase_pairs = [(encoded_phrases[i-1], encoded_phrases[i]) 
                            for i in range(1, len(encoded_phrases))]
            similarity_scores = []
            for pair in phrase_pairs:
                similarity_scores.append(self.calculate_similarity_score(pair))

            # Count tokens
            token_counts = []
            for phrase in phrases:
                token_counts.append(self.count_tokens(phrase))

            # Find threshold
            threshold = self.find_dynamic_threshold(similarity_scores, dynamic_percentile)

            # Create chunks
            chunks = []
            current_chunk = []
            current_chunk_length = 0

            for i, phrase in enumerate(phrases):
                phrase_tokens = token_counts[i]

                if current_chunk_length + phrase_tokens > max_tokens:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [phrase]
                    current_chunk_length = phrase_tokens
                else:
                    current_chunk.append(phrase)
                    current_chunk_length += phrase_tokens

                if (i > 0 and i - 1 < len(similarity_scores) and 
                    similarity_scores[i - 1] < threshold):
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [phrase]
                    current_chunk_length = phrase_tokens

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks, similarity_scores, threshold

        except Exception as e:
            logger.error(f"Error in semantic chunking: {str(e)}")
            return [], [], 0.0

    def get_embedding(self, text: str, model, tokenizer, device) -> np.ndarray:
        """Get embeddings for a text using the specified model."""
        try:
            inputs = tokenizer(text, 
                             return_tensors="pt", 
                             padding="max_length", 
                             truncation=True, 
                             max_length=4096)
            inputs = {key: value.to(device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
            
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()  # Use CLS token embedding
            return cls_embedding
        except Exception as e:
            logger.error(f"Error in getting embeddings: {str(e)}")
            return np.array([])