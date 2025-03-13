import unittest
from pyprosody.emotion_analysis.contextual import ContextualAnalyzer
from pyprosody.text_processing.segmentation import TextSegment

class TestContextualAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = ContextualAnalyzer()

    def test_positive_sentiment(self):
        segment = TextSegment(
            id="test_1",
            text="This movie was absolutely fantastic and inspiring!",
            segment_type="sentence",
            start_pos=0,
            end_pos=48
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertGreater(score.sentiment_score, 0.5)
        self.assertGreater(score.confidence, 0.8)
        self.assertIn("fantastic", score.attention_weights)
        
    def test_negative_sentiment(self):
        segment = TextSegment(
            id="test_2",
            text="The service was terrible and the food was inedible.",
            segment_type="sentence",
            start_pos=0,
            end_pos=50
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertLess(score.sentiment_score, -0.5)
        self.assertGreater(score.confidence, 0.8)
        self.assertIn("terrible", score.attention_weights)
        
    def test_neutral_sentiment(self):
        segment = TextSegment(
            id="test_3",
            text="The train arrives at 3 PM.",
            segment_type="sentence",
            start_pos=0,
            end_pos=25
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertTrue(-0.3 <= score.sentiment_score <= 0.3)
        
    def test_attention_weights(self):
        segment = TextSegment(
            id="test_4",
            text="The very talented actor gave an amazing performance!",
            segment_type="sentence",
            start_pos=0,
            end_pos=52
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertIn("talented", score.attention_weights)
        self.assertIn("amazing", score.attention_weights)
        self.assertGreater(score.attention_weights["amazing"], 
                          score.attention_weights.get("the", 0))

if __name__ == '__main__':
    unittest.main()