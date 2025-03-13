import unittest
from pyprosody.emotion_analysis.sarcasm import SarcasmDetector
from pyprosody.text_processing.segmentation import TextSegment

class TestSarcasmDetector(unittest.TestCase):
    def setUp(self):
        self.detector = SarcasmDetector()
        
    def test_obvious_sarcasm(self):
        segment = TextSegment(
            id="test_1",
            text="Oh, that's just GREAT!!! Exactly what I needed...",
            segment_type="sentence",
            start_pos=0,
            end_pos=45
        )
        score = self.detector.detect_sarcasm(segment)
        
        self.assertGreater(score.probability, 0.7)
        self.assertTrue(score.features.punctuation_patterns)
        
    def test_subtle_sarcasm(self):
        segment = TextSegment(
            id="test_2",
            text="What a absolutely wonderful day to have my car break down.",
            segment_type="sentence",
            start_pos=0,
            end_pos=58
        )
        score = self.detector.detect_sarcasm(segment)
        
        self.assertGreater(score.probability, 0.5)
        self.assertTrue(score.features.sentiment_contrast)
        
    def test_context_based_sarcasm(self):
        main_segment = TextSegment(
            id="test_3",
            text="Perfect timing, as always!",
            segment_type="sentence",
            start_pos=0,
            end_pos=24
        )
        context = [
            TextSegment(
                id="context_1",
                text="The meeting was delayed by two hours.",
                segment_type="sentence",
                start_pos=0,
                end_pos=38
            )
        ]
        score = self.detector.detect_sarcasm(main_segment, context)
        
        self.assertGreater(score.probability, 0.6)
        self.assertTrue(score.features.context_incongruity)
        
    def test_non_sarcastic(self):
        segment = TextSegment(
            id="test_4",
            text="The weather is nice today.",
            segment_type="sentence",
            start_pos=0,
            end_pos=25
        )
        score = self.detector.detect_sarcasm(segment)
        
        self.assertLess(score.probability, 0.3)
        self.assertFalse(any([
            score.features.punctuation_patterns,
            score.features.sentiment_contrast,
            score.features.intensifiers,
            score.features.context_incongruity
        ]))

if __name__ == '__main__':
    unittest.main()