import unittest
import os
from pydub import AudioSegment as PydubSegment
from pyprosody.audio_generation.processor import AudioProcessor, AudioProcessingConfig
from pyprosody.audio_generation.tts import AudioSegment

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.test_dir = "/tmp/pyprosody_test_audio"
        os.makedirs(self.test_dir, exist_ok=True)
        self.processor = AudioProcessor()
        
        # Create test audio segments
        self.segments = [
            AudioSegment(
                segment_id=f"test_{i}",
                audio_path=self._create_test_audio(f"test_{i}.wav", duration=1000),
                duration=1.0,
                sample_rate=44100,
                metadata={}
            )
            for i in range(3)
        ]
        
    def tearDown(self):
        # Clean up test files
        for segment in self.segments:
            if os.path.exists(segment.audio_path):
                os.remove(segment.audio_path)
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def _create_test_audio(self, filename: str, duration: int) -> str:
        """Create a test audio file with given duration in milliseconds."""
        path = os.path.join(self.test_dir, filename)
        audio = PydubSegment.silent(duration=duration)
        audio.export(path, format="wav")
        return path
    
    def test_merge_segments(self):
        output_path = os.path.join(self.test_dir, "merged.wav")
        
        # Merge segments
        result_path = self.processor.merge_segments(self.segments, output_path)
        
        # Verify the output file exists
        self.assertTrue(os.path.exists(result_path))
        
        # Verify the merged audio duration
        merged = PydubSegment.from_wav(result_path)
        expected_duration = sum(s.duration * 1000 for s in self.segments)
        self.assertAlmostEqual(len(merged), expected_duration, delta=200)  # Allow 200ms tolerance for crossfade
        
        # Clean up merged file
        os.remove(output_path)
    
    def test_merge_empty_segments(self):
        with self.assertRaises(ValueError):
            self.processor.merge_segments([], "output.wav")

if __name__ == '__main__':
    unittest.main()