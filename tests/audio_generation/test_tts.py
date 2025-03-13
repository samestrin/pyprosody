import unittest
import os
import shutil
from pyprosody.audio_generation.tts import TTSEngine, TTSConfig, TTSGenerationError
from pyprosody.text_processing.segmentation import TextSegment

class TestTTSEngine(unittest.TestCase):
    def setUp(self):
        self.test_output_dir = "/tmp/pyprosody_test_output"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        self.config = TTSConfig(
            model_name="tts_models/en/ljspeech/glow-tts",
            device="cpu",
            output_format="wav",
            sample_rate=44100
        )
        
        self.tts = TTSEngine(self.config)
        
    def tearDown(self):
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)
    
    def test_basic_speech_generation(self):
        segment = TextSegment(
            id="test_1",
            text="This is a test sentence.",
            segment_type="sentence",
            start_pos=0,
            end_pos=23
        )
        
        audio_segment = self.tts.generate_speech(
            segment,
            self.test_output_dir
        )
        
        self.assertTrue(os.path.exists(audio_segment.audio_path))
        self.assertEqual(audio_segment.segment_id, segment.id)
        self.assertGreater(audio_segment.duration, 0)
        
    def test_prosody_parameters(self):
        segment = TextSegment(
            id="test_2",
            text="Testing prosody parameters.",
            segment_type="sentence",
            start_pos=0,
            end_pos=26
        )
        
        prosody_params = {
            "speed": 1.5,
            "pitch": 1.2,
            "energy": 1.1
        }
        
        audio_segment = self.tts.generate_speech(
            segment,
            self.test_output_dir,
            prosody_params
        )
        
        self.assertTrue(os.path.exists(audio_segment.audio_path))
        self.assertEqual(
            audio_segment.metadata["prosody_params"],
            prosody_params
        )
        
    def test_invalid_text(self):
        segment = TextSegment(
            id="test_3",
            text="",  # Empty text should raise an error
            segment_type="sentence",
            start_pos=0,
            end_pos=0
        )
        
        with self.assertRaises(TTSGenerationError):
            self.tts.generate_speech(
                segment,
                self.test_output_dir
            )

if __name__ == '__main__':
    unittest.main()