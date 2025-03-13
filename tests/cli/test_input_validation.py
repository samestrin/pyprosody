import unittest
from pathlib import Path
import tempfile
from pyprosody.cli.validator import InputValidator, ValidationError

class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.validator = InputValidator()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        Path(self.temp_dir).rmdir()
        
    def test_valid_text_file(self):
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Sample text content")
        
        result = self.validator.validate_input_file(str(test_file))
        self.assertTrue(result)
        
    def test_nonexistent_file(self):
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"
        
        with self.assertRaises(ValidationError) as context:
            self.validator.validate_input_file(str(nonexistent_file))
        self.assertIn("File does not exist", str(context.exception))
        
    def test_empty_file(self):
        empty_file = Path(self.temp_dir) / "empty.txt"
        empty_file.touch()
        
        with self.assertRaises(ValidationError) as context:
            self.validator.validate_input_file(str(empty_file))
        self.assertIn("File is empty", str(context.exception))
        
    def test_invalid_model_name(self):
        with self.assertRaises(ValidationError) as context:
            self.validator.validate_model_name("nonexistent_model")
        self.assertIn("Invalid model name", str(context.exception))
        
    def test_valid_model_name(self):
        result = self.validator.validate_model_name("tts_models/en/ljspeech/glow-tts")
        self.assertTrue(result)