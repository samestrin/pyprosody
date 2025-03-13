from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
from ..text_processing.segmentation import TextSegment
from .lexical import LexicalScore
from .contextual import ContextualScore
from .sarcasm import SarcasmScore
from .pragmatic import PragmaticScore

@dataclass
class EmotionProfile:
    segment_id: str
    text_reference: TextSegment
    basic_sentiment: Dict[str, float]
    complex_emotions: List[Dict[str, float]]
    sarcasm_indicators: Dict[str, float]
    prosody_markers: Dict[str, float]
    metadata: Dict[str, any]

class EmotionCombiner:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'lexical': 0.25,
            'contextual': 0.35,
            'sarcasm': 0.20,
            'pragmatic': 0.20
        }
        
    def combine_emotions(self,
                        segment: TextSegment,
                        lexical: LexicalScore,
                        contextual: ContextualScore,
                        sarcasm: SarcasmScore,
                        pragmatic: PragmaticScore) -> EmotionProfile:
        
        # Calculate basic sentiment
        basic_sentiment = self._combine_sentiment_scores(
            lexical, contextual, sarcasm, pragmatic
        )
        
        # Generate complex emotions
        complex_emotions = self._generate_complex_emotions(
            lexical, contextual, pragmatic
        )
        
        # Process sarcasm indicators
        sarcasm_indicators = {
            'probability': sarcasm.probability,
            'confidence': sarcasm.confidence,
            'feature_count': len(sarcasm.features.feature_descriptions)
        }
        
        # Calculate prosody markers
        prosody_markers = self._calculate_prosody_markers(
            basic_sentiment,
            complex_emotions,
            sarcasm_indicators,
            pragmatic
        )
        
        return EmotionProfile(
            segment_id=segment.id,
            text_reference=segment,
            basic_sentiment=basic_sentiment,
            complex_emotions=complex_emotions,
            sarcasm_indicators=sarcasm_indicators,
            prosody_markers=prosody_markers,
            metadata={
                'timestamp': datetime.utcnow(),
                'model_version': '1.0.0',
                'processing_time': 0.0  # To be set by caller
            }
        )
    
    def _combine_sentiment_scores(self,
                                lexical: LexicalScore,
                                contextual: ContextualScore,
                                sarcasm: SarcasmScore,
                                pragmatic: PragmaticScore) -> Dict[str, float]:
        # Base sentiment calculation
        weighted_sentiment = (
            lexical.polarity * self.weights['lexical'] +
            contextual.sentiment_score * self.weights['contextual']
        )
        
        # Adjust for sarcasm
        if sarcasm.probability > 0.6:
            weighted_sentiment *= -0.5  # Partial sentiment reversal for sarcasm
            
        # Adjust for pragmatic features
        intensity_modifier = pragmatic.emotional_intensity
        weighted_sentiment *= (1.0 + intensity_modifier * 0.5)
        
        return {
            'polarity': max(-1.0, min(1.0, weighted_sentiment)),
            'objectivity': 1.0 - pragmatic.emotional_intensity,
            'confidence': self._calculate_confidence([
                lexical.confidence,
                contextual.confidence,
                sarcasm.confidence
            ])
        }
    
    def _generate_complex_emotions(self,
                                 lexical: LexicalScore,
                                 contextual: ContextualScore,
                                 pragmatic: PragmaticScore) -> List[Dict[str, float]]:
        emotions = []
        
        # Add lexical emotions
        for emotion, score in lexical.emotion_scores.items():
            emotions.append({
                'emotion_type': emotion,
                'intensity': score * self.weights['lexical'],
                'confidence': lexical.confidence
            })
            
        # Adjust intensities based on pragmatic analysis
        intensity_modifier = pragmatic.emotional_intensity
        for emotion in emotions:
            emotion['intensity'] *= (1.0 + intensity_modifier * 0.3)
            emotion['intensity'] = min(1.0, emotion['intensity'])
            
        return emotions
    
    def _calculate_prosody_markers(self,
                                 basic_sentiment: Dict[str, float],
                                 complex_emotions: List[Dict[str, float]],
                                 sarcasm_indicators: Dict[str, float],
                                 pragmatic: PragmaticScore) -> Dict[str, float]:
        # Base speed factor
        speed_factor = 1.0
        if sarcasm_indicators['probability'] > 0.6:
            speed_factor *= 1.2  # Slightly faster for sarcastic content
            
        # Base pitch shift
        pitch_shift = basic_sentiment['polarity'] * 0.3  # -0.3 to +0.3
        
        # Base volume adjustment
        volume_adjust = 1.0 + (pragmatic.emotional_intensity * 0.4)  # 1.0 to 1.4
        
        # Adjust for formality
        if pragmatic.formality_level > 0.7:
            speed_factor *= 0.9  # Slower for formal content
            volume_adjust *= 0.9  # Quieter for formal content
            
        return {
            'speed_factor': max(0.5, min(2.0, speed_factor)),
            'pitch_shift': max(-0.5, min(0.5, pitch_shift)),
            'volume_adjust': max(0.5, min(1.5, volume_adjust)),
            'emphasis_level': pragmatic.emotional_intensity
        }
    
    def _calculate_confidence(self, confidence_scores: List[float]) -> float:
        return sum(confidence_scores) / len(confidence_scores)