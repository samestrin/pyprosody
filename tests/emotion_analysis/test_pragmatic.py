import unittest
from pyprosody.emotion_analysis.pragmatic import PragmaticAnalyzer
from pyprosody.text_processing.segmentation import TextSegment

class TestPragmaticAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = PragmaticAnalyzer()
        
    def test_discourse_markers(self):
        segment = TextSegment(
            id="test_1",
            text="First, we should consider this. However, there are other factors.",
            segment_type="sentence",
            start_pos=0,
            end_pos=63
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertIn('sequence', score.features.discourse_markers)
        self.assertIn('contrast', score.features.discourse_markers)
        
    def test_emphasis_patterns(self):
        segment = TextSegment(
            id="test_2",
            text="This is VERY important!! *Really* significant.",
            segment_type="sentence",
            start_pos=0,
            end_pos=44
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertIn('capitalization', score.features.emphasis_patterns)
        self.assertIn('multiple_punctuation', score.features.emphasis_patterns)
        self.assertIn('italics_markers', score.features.emphasis_patterns)
        
    def test_formality_level(self):
        formal_segment = TextSegment(
            id="test_3",
            text="Moreover, the aforementioned factors consequently lead to...",
            segment_type="sentence",
            start_pos=0,
            end_pos=58
        )
        informal_segment = TextSegment(
            id="test_4",
            text="Well, you know, it's like, pretty much done.",
            segment_type="sentence",
            start_pos=0,
            end_pos=45
        )
        
        formal_score = self.analyzer.analyze_segment(formal_segment)
        informal_score = self.analyzer.analyze_segment(informal_segment)
        
        self.assertGreater(formal_score.formality_level, 
                          informal_score.formality_level)
        
    def test_emotional_intensity(self):
        segment = TextSegment(
            id="test_5",
            text="Indeed, this is absolutely crucial! Really, really important!!!",
            segment_type="sentence",
            start_pos=0,
            end_pos=63
        )
        score = self.analyzer.analyze_segment(segment)
        
        self.assertGreater(score.emotional_intensity, 0.5)
        self.assertGreater(score.certainty_level, 0.7)

if __name__ == '__main__':
    unittest.main()