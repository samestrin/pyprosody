import unittest
from pyprosody.audio_generation.prosody import ProsodyMapper, ProsodyParameters
from pyprosody.emotion_analysis.combiner import EmotionProfile
from pyprosody.text_processing.segmentation import TextSegment

from datetime import datetime

class TestProsodyMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = ProsodyMapper()
        
    def test_positive_emotion_mapping(self):
        profile = EmotionProfile(
            segment_id="test_1",
            text_reference=TextSegment(
                id="test_1",
                text="This is wonderful and happy!",
                segment_type="sentence",
                start_pos=0,
                end_pos=26
            ),
            basic_sentiment={
                'polarity': 0.8,
                'objectivity': 0.2
            },
            complex_emotions=[
                {'type': 'joy', 'intensity': 0.8},
                {'type': 'surprise', 'intensity': 0.3}
            ],
            sarcasm_indicators={
                'probability': 0.1,
                'features': []
            },
            prosody_markers={
                'speed_factor': 1.0,
                'pitch_shift': 0.0,
                'volume_adjust': 1.0
            },
            metadata={
                'attention_weights': {'wonderful': 0.9, 'happy': 0.8},
                'timestamp': datetime.now(),
                'model_version': '1.0'
            }
        )

    def test_negative_emotion_mapping(self):
        profile = EmotionProfile(
            segment_id="test_2",
            text_reference=TextSegment(
                id="test_2",
                text="This is terrible.",
                segment_type="sentence",
                start_pos=0,
                end_pos=16
            ),
            basic_sentiment={
                'polarity': -0.7,
                'objectivity': 0.3
            },
            complex_emotions=[
                {'type': 'sadness', 'intensity': 0.9},
                {'type': 'fear', 'intensity': 0.2}
            ],
            sarcasm_indicators={
                'probability': 0.1,
                'features': []
            },
            prosody_markers={
                'speed_factor': 1.0,
                'pitch_shift': 0.0,
                'volume_adjust': 1.0
            },
            metadata={
                'attention_weights': {'terrible': 0.9},
                'timestamp': datetime.now(),
                'model_version': '1.0'
            }
        )

    def test_sarcastic_content_mapping(self):
        profile = EmotionProfile(
            segment_id="test_3",
            text_reference=TextSegment(
                id="test_3",
                text="This is just great.",
                segment_type="sentence",
                start_pos=0,
                end_pos=18
            ),
            basic_sentiment={
                'polarity': 0.6,
                'objectivity': 0.2
            },
            complex_emotions=[
                {'type': 'surprise', 'intensity': 0.4}
            ],
            sarcasm_indicators={
                'probability': 0.8,
                'features': ['contrast']
            },
            prosody_markers={
                'speed_factor': 1.0,
                'pitch_shift': 0.0,
                'volume_adjust': 1.0
            },
            metadata={
                'attention_weights': {'great': 0.8},
                'timestamp': datetime.now(),
                'model_version': '1.0'
            }
        )
        
        params = self.mapper.map_emotion_to_prosody(profile)
        
        self.assertGreater(params.speed, 1.1)
        self.assertGreater(params.pitch, 1.0)
        self.assertIn('great', params.emphasis_words)

if __name__ == '__main__':
    unittest.main()