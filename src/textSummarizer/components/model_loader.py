from transformers import (
    LongformerTokenizer, LongformerModel,
    BartForConditionalGeneration, BartTokenizer,
    AutoTokenizer, AutoModelForSequenceClassification
)
from sentence_transformers import SentenceTransformer
import torch
from src.textSummarizer.entity import ModelConfig

class ModelLoader:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_models(self):
        longformer = self._load_longformer()
        distilbart = self._load_distilbart()
        sentence_transformer = self._load_sentence_transformer()
        cross_encoder = self._load_cross_encoder()
        
        return {
            "longformer_model": longformer["model"],
            "longformer_tokenizer": longformer["tokenizer"],
            "distilbart_model": distilbart["model"],
            "distilbart_tokenizer": distilbart["tokenizer"],
            "sentence_model": sentence_transformer,
            "rerank_model": cross_encoder["model"],
            "rerank_tokenizer": cross_encoder["tokenizer"]
        }
        
    def _load_longformer(self):
        model_name = self.config.longformer_model_name
        tokenizer = LongformerTokenizer.from_pretrained(model_name)
        model = LongformerModel.from_pretrained(model_name).to(self.device)
        return {"model": model, "tokenizer": tokenizer}
    
    def _load_distilbart(self):
        model_name = self.config.distilbart_model_name
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(self.device)
        return {"model": model, "tokenizer": tokenizer}
    
    def _load_sentence_transformer(self):
        model_name = self.config.sentence_model_name
        return SentenceTransformer(model_name).to(self.device)
    
    def _load_cross_encoder(self):
        model_name = self.config.rerank_model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        model.eval()
        return {"model": model, "tokenizer": tokenizer}