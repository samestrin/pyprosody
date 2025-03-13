import unittest
from datetime import datetime
from pyprosody.emotion_analysis.combiner import EmotionCombiner
from pyprosody.emotion_analysis.lexical import LexicalScore
from pyprosody.emotion_analysis.contextual import ContextualScore
from pyprosody.emotion_analysis.sarcasm import SarcasmScore, SarcasmFeatures
from pyprosody.emotion_analysis.pragmatic import PragmaticScore, DiscourseFeatures
from pyprosody.text_processing.segmentation import TextSegment

class TestEmotionCombiner(unittest.TestCase):
    def setUp(self):
        self.combiner = EmotionCombiner()
        self.segment = TextSegment(
            id="test_1",
            text="This is a test sentence.",
            segment_type="sentence",
            start_pos=0,
            end_pos=23
        )
        
    def test_positive_sentiment_combination(self):
        lexical = LexicalScore(
            afinn_score=0.8,
            positive_score=0.7,
            negative_score=0.1,
            objective_score=0.2,
            compound_score=0.75
        )
        
        contextual = ContextualScore(
            sentiment_score=0.7,
            confidence=0.85,
            attention_weights={'great': 0.8}
        )
        
        sarcasm = SarcasmScore(
            probability=0.1,
            confidence=0.9,
            features=SarcasmFeatures(
                punctuation_patterns=False,
                sentiment_contrast=False,
                intensifiers=False,
                context_incongruity=False,
                feature_descriptions=[]
            )
        )
        
        pragmatic = PragmaticScore(
            features=DiscourseFeatures(
                discourse_markers={},
                emphasis_patterns=[],
                rhetorical_devices=[],
                formality_score=0.5,
                repetition_patterns=[]
            ),
            emotional_intensity=0.6,
            certainty_level=0.8,
            formality_level=0.5
        )
        
        profile = self.combiner.combine_emotions(
            self.segment, lexical, contextual, sarcasm, pragmatic
        )
        
        self.assertGreater(profile.basic_sentiment['polarity'], 0.5)
        self.assertGreater(profile.prosody_markers['pitch_shift'], 0)
        
    def test_sarcastic_content_combination(self):
        lexical = LexicalScore(
            afinn_score=0.8,
            positive_score=0.8,
            negative_score=0.1,
            objective_score=0.3,
            compound_score=0.7
        )
        
        contextual = ContextualScore(
            sentiment_score=0.7,
            confidence=0.85,
            attention_weights={'great': 0.8}
        )
        
        sarcasm = SarcasmScore(
            probability=0.8,
            confidence=0.9,
            features=SarcasmFeatures(
                punctuation_patterns=True,
                sentiment_contrast=True,
                intensifiers=True,
                context_incongruity=True,
                feature_descriptions=['Multiple indicators']
            )
        )
        
        pragmatic = PragmaticScore(
            features=DiscourseFeatures(
                discourse_markers={},
                emphasis_patterns=['capitalization'],
                rhetorical_devices=['rhetorical_question'],
                formality_score=0.3,
                repetition_patterns=[]
            ),
            emotional_intensity=0.8,
            certainty_level=0.7,
            formality_level=0.3
        )
        
        profile = self.combiner.combine_emotions(
            self.segment, lexical, contextual, sarcasm, pragmatic
        )
        
        self.assertLess(profile.basic_sentiment['polarity'], 0)
        self.assertGreater(profile.prosody_markers['speed_factor'], 1.0)
        
    def test_formal_content_combination(self):
        lexical = LexicalScore(
            afinn_score=0.3,
            positive_score=0.4,
            negative_score=0.2,
            objective_score=0.6,
            compound_score=0.3
        )
        
        contextual = ContextualScore(
            sentiment_score=0.4,
            confidence=0.85,
            attention_weights={'accordingly': 0.8}
        )
        
        sarcasm = SarcasmScore(
            probability=0.1,
            confidence=0.9,
            features=SarcasmFeatures(
                punctuation_patterns=False,
                sentiment_contrast=False,
                intensifiers=False,
                context_incongruity=False,
                feature_descriptions=[]
            )
        )
        
        pragmatic = PragmaticScore(
            features=DiscourseFeatures(
                discourse_markers={'formal': 2},
                emphasis_patterns=[],
                rhetorical_devices=[],
                formality_score=0.9,
                repetition_patterns=[]
            ),
            emotional_intensity=0.3,
            certainty_level=0.8,
            formality_level=0.9
        )
        
        profile = self.combiner.combine_emotions(
            self.segment, lexical, contextual, sarcasm, pragmatic
        )
        
        self.assertLess(profile.prosody_markers['speed_factor'], 1.0)
        self.assertLess(profile.prosody_markers['volume_adjust'], 1.0)

if __name__ == '__main__':
    unittest.main()