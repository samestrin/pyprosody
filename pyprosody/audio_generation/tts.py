from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import torch
from TTS.api import TTS
from ..text_processing.segmentation import TextSegment
from ..utils.device import get_optimal_device

@dataclass
class TTSConfig:
    model_name: str = "tts_models/en/ljspeech/glow-tts"
    device: Optional[str] = None  # Now optional, will use optimal device if None
    output_format: str = "wav"
    sample_rate: int = 44100

@dataclass
class AudioSegment:
    segment_id: str
    audio_path: str
    duration: float
    sample_rate: int
    metadata: Dict[str, Any]

class TTSEngine:
    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        self.device = torch.device(self.config.device) if self.config.device else get_optimal_device()
        
        # Initialize TTS
        self.tts = TTS(
            model_name=self.config.model_name,
            progress_bar=False,
            gpu=self.device.type in ["cuda", "mps"]  # Support both CUDA and MPS
        )
        
    def generate_speech(self, 
                       segment: TextSegment, 
                       output_dir: str,
                       prosody_params: Optional[Dict[str, float]] = None) -> AudioSegment:
        """Generate speech from text segment with optional prosody parameters."""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare output path
        output_path = os.path.join(
            output_dir,
            f"{segment.id}.{self.config.output_format}"
        )
        
        # Apply default prosody parameters if none provided
        prosody_params = prosody_params or {
            "speed": 1.0,
            "pitch": 1.0,
            "energy": 1.0
        }
        
        try:
            # Generate speech with prosody parameters
            self.tts.tts_to_file(
                text=segment.text,
                file_path=output_path,
                speed=prosody_params.get("speed", 1.0),
                energy=prosody_params.get("energy", 1.0)
            )
            
            # Get audio duration and metadata
            duration = self._get_audio_duration(output_path)
            
            return AudioSegment(
                segment_id=segment.id,
                audio_path=output_path,
                duration=duration,
                sample_rate=self.config.sample_rate,
                metadata={
                    "prosody_params": prosody_params,
                    "model_name": self.config.model_name,
                    "text_length": len(segment.text)
                }
            )
            
        except Exception as e:
            raise TTSGenerationError(
                f"Failed to generate speech for segment {segment.id}: {str(e)}"
            )
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get the duration of an audio file in seconds."""
        import wave
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            return frames / float(rate)

class TTSGenerationError(Exception):
    """Exception raised for errors during TTS generation."""
    pass