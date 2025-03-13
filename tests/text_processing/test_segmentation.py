import unittest
from pyprosody.text_processing.segmentation import TextSegmenter, TextSegment

class TestTextSegmentation(unittest.TestCase):
    def setUp(self):
        self.segmenter = TextSegmenter()
        
    def test_basic_sentence_segmentation(self):
        text = "This is a test. This is another test."
        segments = self.segmenter.segment(text)
        
        self.assertEqual(len(segments), 2)
        self.assertIsInstance(segments[0], TextSegment)
        self.assertEqual(segments[0].text, "This is a test.")
        self.assertEqual(segments[1].text, "This is another test.")
        
    def test_paragraph_segmentation(self):
        text = "First paragraph.\n\nSecond paragraph."
        segments = self.segmenter.segment(text)
        
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].segment_type, "paragraph")
        self.assertEqual(segments[1].segment_type, "paragraph")
        
    def test_complex_punctuation(self):
        text = "What about this? Yes! No... Maybe?"
        segments = self.segmenter.segment(text)
        
        self.assertEqual(len(segments), 4)
        self.assertEqual(segments[0].text, "What about this?")
        self.assertEqual(segments[1].text, "Yes!")
        self.assertEqual(segments[2].text, "No...")
        self.assertEqual(segments[3].text, "Maybe?")
        
    def test_quoted_text(self):
        text = 'He said "This is a quote." Then continued.'
        segments = self.segmenter.segment(text)
        
        self.assertEqual(len(segments), 2)
        self.assertIn("quote", segments[0].text)
        self.assertEqual(segments[1].text, "Then continued.")
        
    def test_segment_positions(self):
        text = "First. Second."
        segments = self.segmenter.segment(text)
        
        self.assertEqual(segments[0].start_pos, 0)
        self.assertEqual(segments[0].end_pos, 6)
        self.assertEqual(segments[1].start_pos, 7)
        self.assertEqual(segments[1].end_pos, 14)
        
    def test_empty_input(self):
        with self.assertRaises(ValueError):
            self.segmenter.segment("")
            
    def test_special_characters(self):
        text = "Test with émoji 😊! And some UTF-8 characters."
        segments = self.segmenter.segment(text)
        
        self.assertEqual(len(segments), 2)
        self.assertIn("émoji", segments[0].text)
        self.assertIn("😊", segments[0].text)