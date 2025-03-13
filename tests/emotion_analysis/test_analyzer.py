import unittest
from datetime import datetime
from pyprosody.emotion_analysis.analyzer import EmotionAnalyzer
from pyprosody.text_processing.segmentation import TextSegment

class TestEmotionAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = EmotionAnalyzer()
        self.test_segment = TextSegment(
            id="test_1",
            text="I am really happy about this wonderful day!",
            segment_type="sentence",
            start_pos=0,
            end_pos=41
        )

    def test_analyze_positive_emotion(self):
        profile = self.analyzer.analyze(self.test_segment)
        
        # Check basic structure
        self.assertEqual(profile.segment_id, "test_1")
        self.assertEqual(profile.text_reference, self.test_segment)
        
        # Check sentiment
        self.assertGreater(profile.basic_sentiment['polarity'], 0)
        self.assertIn('objectivity', profile.basic_sentiment)
        
        # Check complex emotions
        self.assertTrue(any(e['type'] == 'joy' for e in profile.complex_emotions))
        
        # Check metadata
        self.assertIsInstance(profile.metadata['timestamp'], datetime)
        self.assertIn('model_version', profile.metadata)
        self.assertIn('processing_time', profile.metadata)

    def test_analyze_sarcasm(self):
        sarcastic_segment = TextSegment(
            id="test_2",
            text="Oh great, another wonderful meeting...",
            segment_type="sentence",
            start_pos=0,
            end_pos=35
        )
        
        profile = self.analyzer.analyze(sarcastic_segment)
        
        self.assertGreater(profile.sarcasm_indicators['probability'], 0.5)
        self.assertTrue(len(profile.sarcasm_indicators['features']) > 0)

    def test_analyze_empty_text(self):
        empty_segment = TextSegment(
            id="test_3",
            text="",
            segment_type="sentence",
            start_pos=0,
            end_pos=0
        )
        
        with self.assertRaises(ValueError):
            self.analyzer.analyze(empty_segment)

if __name__ == '__main__':
    unittest.main()