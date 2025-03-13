from dataclasses import dataclass
from typing import List, Dict, Optional
import re
from ..text_processing.segmentation import TextSegment

@dataclass
class SarcasmFeatures:
    punctuation_patterns: bool
    sentiment_contrast: bool
    intensifiers: bool
    context_incongruity: bool
    feature_descriptions: List[str]

@dataclass
class SarcasmScore:
    probability: float
    features: SarcasmFeatures
    confidence: float

class SarcasmDetector:
    def __init__(self):
        self.intensifier_words = {
            'very', 'really', 'absolutely', 'totally', 'completely',
            'utterly', 'literally', 'obviously', 'clearly', 'surely'
        }
        
        self.positive_phrases = {
            'great', 'wonderful', 'amazing', 'fantastic', 'brilliant',
            'perfect', 'excellent', 'outstanding', 'superb', 'terrific'
        }
        
    def detect_sarcasm(self, segment: TextSegment, 
                      context_segments: Optional[List[TextSegment]] = None) -> SarcasmScore:
        text = segment.text
        features = []
        
        # Check punctuation patterns
        punctuation_pattern = self._check_punctuation_patterns(text)
        if punctuation_pattern:
            features.append("Unusual punctuation patterns detected")
            
        # Check for intensifiers
        intensifiers = self._check_intensifiers(text)
        if intensifiers:
            features.append("Excessive use of intensifiers")
            
        # Check for sentiment contrast
        sentiment_contrast = self._check_sentiment_contrast(text)
        if sentiment_contrast:
            features.append("Contrasting sentiment indicators")
            
        # Check context incongruity
        context_incongruity = False
        if context_segments:
            context_incongruity = self._check_context_incongruity(segment, context_segments)
            if context_incongruity:
                features.append("Contextual incongruity detected")
        
        # Calculate probability based on features
        feature_count = sum([
            punctuation_pattern,
            intensifiers,
            sentiment_contrast,
            context_incongruity
        ])
        
        # Base probability on feature count with diminishing returns
        probability = min(0.9, (feature_count / 4) * 0.8)
        
        # Confidence based on feature diversity and strength
        confidence = min(0.95, (len(features) / 4) * 0.85)
        
        return SarcasmScore(
            probability=probability,
            features=SarcasmFeatures(
                punctuation_patterns=punctuation_pattern,
                sentiment_contrast=sentiment_contrast,
                intensifiers=intensifiers,
                context_incongruity=context_incongruity,
                feature_descriptions=features
            ),
            confidence=confidence
        )
    
    def _check_punctuation_patterns(self, text: str) -> bool:
        # Check for repeated punctuation or mixed punctuation
        patterns = [
            r'[!?]{2,}',          # Multiple ! or ?
            r'[!?][.!?]+',        # Mixed punctuation
            r'\.{3,}',            # Ellipsis
            r'(!|\?)\s*\1{1,}'    # Repeated ! or ? with possible spaces
        ]
        return any(re.search(pattern, text) for pattern in patterns)
    
    def _check_intensifiers(self, text: str) -> bool:
        words = text.lower().split()
        intensifier_count = sum(1 for word in words if word in self.intensifier_words)
        return intensifier_count >= 2
    
    def _check_sentiment_contrast(self, text: str) -> bool:
        # Check for positive phrases in negative contexts or vice versa
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_phrases)
        has_negation = any(neg in text.lower() for neg in ['not', "n't", 'never'])
        return positive_count > 0 and has_negation
    
    def _check_context_incongruity(self, 
                                 segment: TextSegment, 
                                 context_segments: List[TextSegment]) -> bool:
        # Simple check for tonal shift between segments
        current_words = set(segment.text.lower().split())
        context_words = set()
        
        for ctx_segment in context_segments:
            context_words.update(ctx_segment.text.lower().split())
        
        # Check for significant vocabulary shift
        common_words = current_words.intersection(context_words)
        return len(common_words) < len(current_words) * 0.3