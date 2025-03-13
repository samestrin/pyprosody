import unittest
from pyprosody.emotion_analysis.lexical import LexicalAnalyzer
from pyprosody.text_processing.segmentation import TextSegment

class TestLexicalAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = LexicalAnalyzer()
        
    def test_positive_sentiment(self):
        segment = TextSegment(
            id="test_1",
            text="I am very happy and excited about this wonderful day!",
            segment_type="sentence",
            start_pos=0,
            end_pos=50
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertGreater(score.afinn_score, 0)
        self.assertGreater(score.positive_score, score.negative_score)
        self.assertGreater(score.compound_score, 0)
        
    def test_negative_sentiment(self):
        segment = TextSegment(
            id="test_2",
            text="I am very sad and disappointed about this terrible situation.",
            segment_type="sentence",
            start_pos=0,
            end_pos=50
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertLess(score.afinn_score, 0)
        self.assertGreater(score.negative_score, score.positive_score)
        self.assertLess(score.compound_score, 0)
        
    def test_neutral_sentiment(self):
        segment = TextSegment(
            id="test_3",
            text="The cat sat on the mat.",
            segment_type="sentence",
            start_pos=0,
            end_pos=22
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertAlmostEqual(score.afinn_score, 0, delta=0.1)
        self.assertAlmostEqual(score.compound_score, 0, delta=0.2)
        self.assertGreater(score.objective_score, 0.5)
        
    def test_mixed_sentiment(self):
        segment = TextSegment(
            id="test_4",
            text="While I love the beautiful weather, I hate being stuck inside.",
            segment_type="sentence",
            start_pos=0,
            end_pos=60
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertTrue(-2 <= score.compound_score <= 2)
        self.assertGreater(score.positive_score, 0)
        self.assertGreater(score.negative_score, 0)
        
    def test_empty_text(self):
        segment = TextSegment(
            id="test_5",
            text="",
            segment_type="sentence",
            start_pos=0,
            end_pos=0
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertEqual(score.afinn_score, 0)
        self.assertEqual(score.positive_score, 0)
        self.assertEqual(score.negative_score, 0)
        self.assertEqual(score.objective_score, 1.0)
        self.assertEqual(score.compound_score, 0)

if __name__ == '__main__':
    unittest.main()