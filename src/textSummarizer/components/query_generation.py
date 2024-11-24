from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
from collections import Counter
from gensim import corpora
from gensim.models import LdaModel
from typing import List, Tuple, Dict
from src.textSummarizer.logging import logger

class QueryGeneration:
    def __init__(self):
        # self.config = config
        self.logger = logger

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for keyword extraction."""
        try:
            tokens = word_tokenize(text.lower())
            return [token for token in tokens if token not in STOPWORDS and token.isalnum()]
        except Exception as e:
            self.logger.error(f"Error in text preprocessing: {str(e)}")
            return []

    def get_most_common_words(self, chunks: List[str], n: int = 10) -> List[str]:
        """Extract most common words from chunks."""
        try:
            all_words = [word for chunk in chunks 
                        for word in self.preprocess_text(chunk)]
            return [word for word, _ in Counter(all_words).most_common(n)]
        except Exception as e:
            self.logger.error(f"Error in extracting common words: {str(e)}")
            return []

    def extract_topics(self, chunks: List[str], num_topics: int = 5) -> List[Tuple[int, List[Tuple[str, float]]]]:
        """Extract topics using LDA."""
        try:
            preprocessed_chunks = [self.preprocess_text(chunk) for chunk in chunks]
            dictionary = corpora.Dictionary(preprocessed_chunks)
            corpus = [dictionary.doc2bow(chunk) for chunk in preprocessed_chunks]
            
            lda_model = LdaModel(corpus=corpus, 
                                id2word=dictionary, 
                                num_topics=num_topics)
            
            topics = lda_model.print_topics(num_words=10)
            return [(topic_id, 
                     [(word.split('*')[1].strip('\"'), 
                       float(word.split('*')[0])) 
                      for word in topic_words.split('+')]) 
                    for topic_id, topic_words in topics]
        except Exception as e:
            self.logger.error(f"Error in topic extraction: {str(e)}")
            return []

    def generate_domain_specific_query(self, 
                                     chunks: List[str], 
                                     keywords: List[str], 
                                     num_topics: int = 3) -> str:
        """Generate a domain-specific query based on topics and keywords."""
        try:
            # Extract topics
            topics = self.extract_topics(chunks, num_topics)
            topic_words = [word for _, topic_terms in topics 
                          for word, _ in topic_terms]
            
            # Get common words
            common_words = self.get_most_common_words(chunks)
            
            # Combine keywords and context
            query_elements_context = list(set(topic_words + common_words))
            
            # Construct query
            main_topics = ', '.join(keywords)
            context = ' '.join(query_elements_context)
            
            return f"Summarize the key points related to {main_topics} in the context of {context}."
        except Exception as e:
            self.logger.error(f"Error in query generation: {str(e)}")
            return ""