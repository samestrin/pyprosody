from typing import Dict, Optional
import logging
from pathlib import Path

from ..text_processing.reader import TextReader
from ..text_processing.segmentation import TextSegmenter
from ..emotion_analysis.analyzer import EmotionAnalyzer 
from ..audio_generation.tts import TTSEngine
from ..audio_generation.prosody import ProsodyMapper
from ..audio_generation.processor import AudioProcessor

from ..utils.exceptions import PipelineError, TextProcessingError, EmotionAnalysisError, AudioGenerationError
from ..utils.logging import get_logger, LogContext

class Pipeline:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = get_logger(__name__)
        
        try:
            # Initialize components
            self.text_reader = TextReader()
            self.text_segmenter = TextSegmenter()
            self.emotion_analyzer = EmotionAnalyzer()
            self.tts_engine = TTSEngine()
            self.prosody_mapper = ProsodyMapper()
            self.audio_processor = AudioProcessor()
        except Exception as e:
            raise PipelineError(f"Failed to initialize pipeline: {str(e)}") from e
        
    def process(self, input_path: str, output_path: str) -> Dict:
        with LogContext(self.logger, input_path=input_path, output_path=output_path):
            try:
                # Read and segment text
                self.logger.info("Processing input file", extra={'file': input_path})
                text = self.text_reader.read_file(input_path)
                segments = self.text_segmenter.segment(text)
                
                # Analyze emotions
                self.logger.info("Analyzing emotions", extra={'segment_count': len(segments)})
                emotion_profiles = []
                for segment in segments:
                    profile = self.emotion_analyzer.analyze(segment)
                    emotion_profiles.append(profile)
                
                # Generate audio segments
                self.logger.info("Generating audio")
                audio_segments = []
                for segment, profile in zip(segments, emotion_profiles):
                    prosody_params = self.prosody_mapper.map_emotion_to_prosody(profile)
                    audio = self.tts_engine.generate_speech(
                        segment, 
                        output_dir=str(Path(output_path).parent),
                        prosody_params=prosody_params.__dict__
                    )
                    audio_segments.append(audio)
                
                # Merge audio segments
                self.logger.info("Merging audio segments")
                final_path = self.audio_processor.merge_segments(audio_segments, output_path)
                
                return {
                    "success": True,
                    "output_path": final_path,
                    "stats": {
                        "segments_processed": len(segments),
                        "total_duration": sum(seg.duration for seg in audio_segments)
                    }
                }
                
            except TextProcessingError as e:
                self.logger.error("Text processing failed", exc_info=True)
                return {"success": False, "error": str(e), "stats": {}}
            except EmotionAnalysisError as e:
                self.logger.error("Emotion analysis failed", exc_info=True)
                return {"success": False, "error": str(e), "stats": {}}
            except AudioGenerationError as e:
                self.logger.error("Audio generation failed", exc_info=True)
                return {"success": False, "error": str(e), "stats": {}}
            except Exception as e:
                self.logger.error("Unexpected error in pipeline", exc_info=True)
                return {"success": False, "error": str(e), "stats": {}}