import unittest
import os
from pathlib import Path
from pyprosody.pipeline.main import Pipeline

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline()
        self.test_dir = Path("/tmp/pyprosody_test")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a test input file
        self.input_path = self.test_dir / "test_input.txt"
        with open(self.input_path, "w") as f:
            f.write("This is a test sentence. It has multiple emotions!")
        
        self.output_path = self.test_dir / "test_output.wav"
    
    def tearDown(self):
        # Clean up test files
        if self.input_path.exists():
            self.input_path.unlink()
        if self.output_path.exists():
            self.output_path.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
    
    def test_pipeline_success(self):
        result = self.pipeline.process(
            str(self.input_path),
            str(self.output_path)
        )
        
        self.assertTrue(result["success"])
        self.assertTrue(os.path.exists(result["output_path"]))
        self.assertGreater(result["stats"]["segments_processed"], 0)
        self.assertGreater(result["stats"]["total_duration"], 0)
    
    def test_pipeline_invalid_input(self):
        result = self.pipeline.process(
            "nonexistent.txt",
            str(self.output_path)
        )
        
        self.assertFalse(result["success"])
        self.assertIn("error", result)

if __name__ == '__main__':
    unittest.main()