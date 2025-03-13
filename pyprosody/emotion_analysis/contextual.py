from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from ..text_processing.segmentation import TextSegment
from ..utils.device import get_optimal_device

@dataclass
class ContextualScore:
    sentiment_score: float  # Range: -1.0 to 1.0
    confidence: float      # Range: 0.0 to 1.0
    attention_weights: Dict[str, float]

class ContextualAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.device = get_optimal_device()
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def analyze_segment(self, segment: TextSegment, context_window: int = 2) -> ContextualScore:
        with torch.no_grad():
            # Prepare input
            inputs = self.tokenizer(
                segment.text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get model outputs
            outputs = self.model(**inputs, output_attentions=True)
            
            # Process logits
            logits = outputs.logits.squeeze()
            probs = torch.softmax(logits, dim=0)
            
            # Convert to sentiment score (-1 to 1)
            sentiment_score = float((probs[1] - probs[0]).cpu().numpy())
            confidence = float(torch.max(probs).cpu().numpy())
            
            # Process attention weights
            attention_weights = self._process_attention_weights(
                inputs['input_ids'][0],
                outputs.attentions[-1][0][-1].mean(dim=0)
            )

            return ContextualScore(
                sentiment_score=sentiment_score,
                confidence=confidence,
                attention_weights=attention_weights
            )

    def _process_attention_weights(self, input_ids: torch.Tensor, attention: torch.Tensor) -> Dict[str, float]:
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        attention = attention.cpu().numpy()
        
        # Create token-attention mapping
        token_attention = {}
        for token, weight in zip(tokens, attention):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                # Remove ## from wordpiece tokens
                clean_token = token.replace('##', '')
                if clean_token in token_attention:
                    token_attention[clean_token] = max(token_attention[clean_token], float(weight))
                else:
                    token_attention[clean_token] = float(weight)
        
        return token_attention