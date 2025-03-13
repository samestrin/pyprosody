import re
from typing import List
from .segmentation import TextSegment

class TextPreprocessor:
    def __init__(self):
        self.special_chars_pattern = re.compile(r'[^a-zA-Z0-9\s.,!?\'"-]')
        self.multiple_spaces = re.compile(r'\s+')
        self.multiple_periods = re.compile(r'\.{2,}')

    def preprocess_segments(self, segments: List[TextSegment]) -> List[TextSegment]:
        processed_segments = []
        
        for segment in segments:
            processed_text = self._normalize_text(segment.text)
            
            # Create new segment with processed text while preserving metadata
            processed_segment = TextSegment(
                id=segment.id,
                text=processed_text,
                segment_type=segment.segment_type,
                start_pos=segment.start_pos,
                end_pos=segment.end_pos,
                parent_id=segment.parent_id
            )
            processed_segments.append(processed_segment)
            
        return processed_segments

    def _normalize_text(self, text: str) -> str:
        # Preserve ellipsis as it might indicate sarcasm
        text = self.multiple_periods.sub('...', text)
        
        # Handle special characters while preserving meaningful punctuation
        text = self.special_chars_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self.multiple_spaces.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text