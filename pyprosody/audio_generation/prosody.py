from dataclasses import dataclass
from typing import Dict, List, Optional
from ..emotion_analysis.combiner import EmotionProfile
from ..text_processing.segmentation import TextSegment

@dataclass
class ProsodyParameters:
    speed: float = 1.0      # Range: 0.5 to 2.0
    pitch: float = 0.0      # Range: -20.0 to 20.0 (semitones)
    energy: float = 1.0     # Range: 0.5 to 2.0
    emphasis_words: List[str] = None

class ProsodyMapper:
    def __init__(self):
        self.emotion_speed_map = {
            'joy': 1.2,
            'sadness': 0.8,
            'anger': 1.3,
            'fear': 1.1,
            'surprise': 1.15
        }
        
        self.emotion_pitch_map = {
            'joy': 2.0,
            'sadness': -3.0,
            'anger': 4.0,
            'fear': 3.0,
            'surprise': 5.0
        }
        
        self.emotion_energy_map = {
            'joy': 1.2,
            'sadness': 0.8,
            'anger': 1.4,
            'fear': 1.1,
            'surprise': 1.2
        }
    
    def map_emotion_to_prosody(self, profile: EmotionProfile) -> ProsodyParameters:
        params = ProsodyParameters()
        
        # Adjust for basic sentiment
        sentiment_intensity = abs(profile.basic_sentiment['polarity'])
        
        # Speed adjustments
        base_speed = 1.0
        if profile.basic_sentiment['polarity'] > 0:
            base_speed *= 1.1
        elif profile.basic_sentiment['polarity'] < 0:
            base_speed *= 0.9
            
        # Adjust for sarcasm
        if profile.sarcasm_indicators['probability'] > 0.7:
            base_speed *= 1.15
            params.pitch += 2.0
            
        # Apply emotion-specific modifiers
        for emotion in profile.complex_emotions:
            emotion_type = emotion['type']
            intensity = emotion['intensity']
            
            if emotion_type in self.emotion_speed_map:
                base_speed += (self.emotion_speed_map[emotion_type] - 1.0) * intensity
                params.pitch += self.emotion_pitch_map[emotion_type] * intensity
                params.energy *= (1.0 + (self.emotion_energy_map[emotion_type] - 1.0) * intensity)
        
        # Clamp values to valid ranges
        params.speed = max(0.5, min(2.0, base_speed))
        params.pitch = max(-20.0, min(20.0, params.pitch))
        params.energy = max(0.5, min(2.0, params.energy))
        
        # Extract emphasis words from attention weights
        attention_weights = profile.metadata.get('attention_weights', {})
        params.emphasis_words = [
            word for word, weight in attention_weights.items()
            if weight > 0.7
        ]
        
        return params