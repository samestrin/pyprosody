import unittest
import tempfile
from pathlib import Path
import json
from datetime import datetime

from pyprosody.pipeline.main import Pipeline
from pyprosody.utils.exceptions import PipelineError

class TestPipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()
        self.test_dir = tempfile.mkdtemp()
        self.sample_stories = {
            "basic": "This is a happy story. The sun was shining brightly!",
            "complex": """The old man smiled sadly. "Life is full of surprises," he said 
                        with a knowing look. But deep down, he felt a surge of hope.""",
            "emotional": """Sarah was ecstatic! She had finally achieved her dream. 
                          However, the victory felt bittersweet... her mentor couldn't 
                          be there to see it."""
        }
        
    def tearDown(self):
        # Clean up test files
        for file in Path(self.test_dir).glob("*"):
            file.unlink()
        Path(self.test_dir).rmdir()
    
    def create_test_file(self, content: str) -> Path:
        test_file = Path(self.test_dir) / "test_input.txt"
        test_file.write_text(content)
        return test_file
    
    def test_basic_story_processing(self):
        input_file = self.create_test_file(self.sample_stories["basic"])
        output_file = Path(self.test_dir) / "output_basic.wav"
        
        result = self.pipeline.process(str(input_file), str(output_file))
        
        self.assertTrue(result["success"])
        self.assertTrue(output_file.exists())
        self.assertGreater(output_file.stat().st_size, 0)
        
    def test_complex_emotional_story(self):
        input_file = self.create_test_file(self.sample_stories["complex"])
        output_file = Path(self.test_dir) / "output_complex.wav"
        
        result = self.pipeline.process(str(input_file), str(output_file))
        
        self.assertTrue(result["success"])
        self.assertGreater(result["stats"]["segments_processed"], 2)
        self.assertTrue(output_file.exists())
        
    def test_emotion_profile_consistency(self):
        input_file = self.create_test_file(self.sample_stories["emotional"])
        output_file = Path(self.test_dir) / "output_emotional.wav"
        
        with self.pipeline.logger.capture_logs() as logs:
            result = self.pipeline.process(str(input_file), str(output_file))
            
        # Verify emotion analysis logs
        emotion_logs = [log for log in logs if "Analyzing emotions" in log["message"]]
        self.assertGreater(len(emotion_logs), 0)
        
        # Check for emotional transitions in the audio
        self.assertTrue(result["success"])
        self.assertGreater(result["stats"]["total_duration"], 0)
    
    def test_pipeline_error_recovery(self):
        # Test with malformed input
        input_file = self.create_test_file("") # Empty file
        output_file = Path(self.test_dir) / "output_error.wav"
        
        result = self.pipeline.process(str(input_file), str(output_file))
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    def test_resource_cleanup(self):
        input_file = self.create_test_file(self.sample_stories["basic"])
        output_file = Path(self.test_dir) / "output_cleanup.wav"
        
        # Force resource cleanup by simulating an error
        with self.assertRaises(PipelineError):
            with self.pipeline.logger.capture_logs():
                self.pipeline.process(
                    str(input_file), 
                    str(output_file),
                    config={"force_error": True}
                )
        
        # Verify no temporary files are left
        temp_files = list(Path(self.test_dir).glob("*.tmp"))
        self.assertEqual(len(temp_files), 0)

if __name__ == '__main__':
    unittest.main()