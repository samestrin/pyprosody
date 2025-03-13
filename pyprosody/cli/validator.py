from pathlib import Path
from typing import List

class ValidationError(Exception):
    """Custom exception for input validation errors."""
    pass

class InputValidator:
    def __init__(self):
        # List of supported TTS models
        self.supported_models = [
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/ljspeech/tacotron2",
            "tts_models/en/vctk/vits"
        ]

    def validate_input_file(self, file_path: str) -> bool:
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
            
        if not path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
            
        if path.stat().st_size == 0:
            raise ValidationError(f"File is empty: {file_path}")
            
        if path.suffix.lower() != '.txt':
            raise ValidationError(f"File must be a .txt file: {file_path}")
            
        return True

    def validate_model_name(self, model_name: str) -> bool:
        if model_name not in self.supported_models:
            raise ValidationError(f"Invalid model name: {model_name}. Supported models: {', '.join(self.supported_models)}")
            
        return True