from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TextProcessingConfig:
    encoding: str = 'utf-8'
    max_file_size: int = 10_000_000  # 10MB
    retry_attempts: int = 3

@dataclass
class SegmentationMetadata:
    timestamp: datetime
    file_path: str
    total_chars: int
    total_segments: int
    processing_time: float

from .segmentation import TextSegmenter, TextSegment
from .preprocessor import TextPreprocessor

class TextReader:
    def __init__(self, config: TextProcessingConfig | None = None) -> None:
        self.config = config or TextProcessingConfig()
        self.segmenter = TextSegmenter()
        self.preprocessor = TextPreprocessor()

    def process_text(self, file_path: str) -> tuple[list[TextSegment], SegmentationMetadata]:
        content, basic_metadata = self.read_file(file_path)
        segments = self.segmenter.segment_text(content)
        processed_segments = self.preprocessor.preprocess_segments(segments)
        
        metadata = SegmentationMetadata(
            timestamp=basic_metadata.timestamp,
            file_path=basic_metadata.file_path,
            total_chars=basic_metadata.total_chars,
            total_segments=len(processed_segments),
            processing_time=(datetime.utcnow() - basic_metadata.timestamp).total_seconds()
        )
        
        return processed_segments, metadata

    def read_file(self, file_path: str) -> tuple[str, SegmentationMetadata]:
        path = Path(file_path)
        start_time = datetime.utcnow()
        
        try:
            if path.stat().st_size > self.config.max_file_size:
                raise ValueError(f"File size exceeds maximum limit of {self.config.max_file_size} bytes")
            
            with path.open('r', encoding=self.config.encoding) as file:
                content = file.read()
            
            metadata = SegmentationMetadata(
                timestamp=start_time,
                file_path=str(path),
                total_chars=len(content),
                total_segments=1,  # Will be updated after segmentation
                processing_time=(datetime.utcnow() - start_time).total_seconds()
            )
            
            return content, metadata
            
        except FileNotFoundError:
            raise TextProcessingError(f"File not found: {file_path}")
        except PermissionError:
            raise TextProcessingError(f"Permission denied: {file_path}")
        except UnicodeDecodeError:
            raise TextProcessingError(f"File encoding error: {file_path}")
        except Exception as e:
            raise TextProcessingError(f"Error reading file: {str(e)}")

class TextProcessingError(Exception):
    """Custom exception for text processing errors"""
    pass