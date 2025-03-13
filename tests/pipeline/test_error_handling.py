import unittest
import logging
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from pyprosody.pipeline.main import Pipeline
from pyprosody.utils.exceptions import (
    TextProcessingError,
    EmotionAnalysisError,
    AudioGenerationError,
    PipelineError
)

class TestPipelineErrorHandling(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()
        self.test_input = "/tmp/test_input.txt"
        self.test_output = "/tmp/test_output.wav"
        
        # Create test input file
        Path(self.test_input).write_text("Test content")
    
    def tearDown(self):
        # Clean up test files
        for path in [self.test_input, self.test_output]:
            if Path(path).exists():
                Path(path).unlink()
    
    @patch('pyprosody.pipeline.main.TextReader')
    def test_text_processing_error(self, mock_reader):
        mock_reader.return_value.read_file.side_effect = TextProcessingError("Failed to read file")
        
        result = self.pipeline.process(self.test_input, self.test_output)
        
        self.assertFalse(result["success"])
        self.assertIn("Failed to read file", result["error"])
        self.assertEqual(result["stats"], {})
    
    @patch('pyprosody.pipeline.main.EmotionAnalyzer')
    def test_emotion_analysis_error(self, mock_analyzer):
        mock_analyzer.return_value.analyze.side_effect = EmotionAnalysisError("Failed to analyze emotions")
        
        result = self.pipeline.process(self.test_input, self.test_output)
        
        self.assertFalse(result["success"])
        self.assertIn("Failed to analyze emotions", result["error"])
        self.assertEqual(result["stats"], {})
    
    @patch('pyprosody.pipeline.main.TTSEngine')
    def test_audio_generation_error(self, mock_tts):
        mock_tts.return_value.generate_speech.side_effect = AudioGenerationError("Failed to generate audio")
        
        result = self.pipeline.process(self.test_input, self.test_output)
        
        self.assertFalse(result["success"])
        self.assertIn("Failed to generate audio", result["error"])
        self.assertEqual(result["stats"], {})
    
    def test_pipeline_initialization_error(self):
        with patch('pyprosody.pipeline.main.TextReader', side_effect=Exception("Init failed")):
            with self.assertRaises(PipelineError) as context:
                Pipeline()
            self.assertIn("Failed to initialize pipeline", str(context.exception))
    
    @patch('pyprosody.utils.logging.logging.FileHandler')
    def test_log_output_format(self, mock_handler):
        captured_logs = []
        mock_handler.return_value.handle = lambda record: captured_logs.append(record)
        
        self.pipeline.process(self.test_input, self.test_output)
        
        for record in captured_logs:
            log_data = json.loads(record.msg)
            self.assertIn('timestamp', log_data)
            self.assertIn('level', log_data)
            self.assertIn('module', log_data)
            self.assertIn('thread_id', log_data)
            self.assertIn('message', log_data)
            self.assertIn('context', log_data)

if __name__ == '__main__':
    unittest.main()