from dataclasses import dataclass
from typing import List, Optional
from pydub import AudioSegment as PydubSegment
import os
from .tts import AudioSegment

@dataclass
class AudioProcessingConfig:
    crossfade_duration: float = 100  # milliseconds
    output_format: str = "wav"
    sample_rate: int = 44100
    channels: int = 1
    normalize: bool = True

class AudioProcessor:
    def __init__(self, config: Optional[AudioProcessingConfig] = None):
        self.config = config or AudioProcessingConfig()
    
    def merge_segments(self, segments: List[AudioSegment], output_path: str) -> str:
        """Merge multiple audio segments into a single file with smooth transitions."""
        if not segments:
            raise ValueError("No audio segments provided")
            
        # Load the first segment
        merged = PydubSegment.from_wav(segments[0].audio_path)
        
        # Add subsequent segments with crossfade
        for i in range(1, len(segments)):
            next_segment = PydubSegment.from_wav(segments[i].audio_path)
            
            # Apply crossfade
            merged = merged.append(next_segment, 
                                 crossfade=int(self.config.crossfade_duration))
        
        # Normalize if configured
        if self.config.normalize:
            merged = merged.normalize()
            
        # Export the final audio
        merged.export(
            output_path,
            format=self.config.output_format,
            parameters=[
                "-ar", str(self.config.sample_rate),
                "-ac", str(self.config.channels)
            ]
        )
        
        return output_path