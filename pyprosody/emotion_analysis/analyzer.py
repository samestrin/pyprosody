from typing import List, Optional
from datetime import datetime

from .lexical import LexicalAnalyzer
from .contextual import ContextualAnalyzer
from .sarcasm import SarcasmDetector
from .pragmatic import PragmaticAnalyzer
from .combiner import EmotionProfile
from ..text_processing.segmentation import TextSegment

class EmotionAnalyzer:
    def __init__(self):
        self.lexical_analyzer = LexicalAnalyzer()
        self.contextual_analyzer = ContextualAnalyzer()
        self.sarcasm_detector = SarcasmDetector()
        self.pragmatic_analyzer = PragmaticAnalyzer()
    
    def analyze(self, segment: TextSegment) -> EmotionProfile:
        # Run all analysis components
        lexical_score = self.lexical_analyzer.analyze(segment.text)
        contextual_score = self.contextual_analyzer.analyze(segment.text)
        sarcasm_score = self.sarcasm_detector.detect(segment.text)
        pragmatic_score = self.pragmatic_analyzer.analyze(segment.text)
        
        # Create emotion profile
        return EmotionProfile(
            segment_id=segment.id,
            text_reference=segment,
            basic_sentiment={
                'polarity': contextual_score.sentiment,
                'objectivity': contextual_score.objectivity
            },
            complex_emotions=[
                {'type': emotion, 'intensity': score}
                for emotion, score in contextual_score.emotions.items()
            ],
            sarcasm_indicators={
                'probability': sarcasm_score.probability,
                'features': sarcasm_score.features
            },
            prosody_markers={
                'speed_factor': 1.0,
                'pitch_shift': 0.0,
                'volume_adjust': 1.0,
                'emphasis_points': pragmatic_score.emphasis_points
            },
            metadata={
                'timestamp': datetime.now(),
                'model_version': '1.0',
                'processing_time': contextual_score.processing_time
            }
        )