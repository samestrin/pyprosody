import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any

class CLIHandler:
    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="PyProsody: Convert text to emotionally-aware speech",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        
        parser.add_argument(
            "input_file",
            type=str,
            help="Path to the input text file"
        )
        
        parser.add_argument(
            "--model",
            type=str,
            default="default",
            help="Name of the emotion analysis model to use"
        )
        
        parser.add_argument(
            "--output",
            type=str,
            help="Path to save the output audio file"
        )
        
        return parser

    def validate_input(self, file_path: str) -> bool:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {file_path}")
        if not path.suffix == '.txt':
            raise ValueError(f"Input file must be a .txt file: {file_path}")
        return True

    def validate_model(self, model_name: str) -> bool:
        # TODO: Implement model validation against available models
        valid_models = ["default", "basic", "advanced"]
        if model_name not in valid_models:
            raise ValueError(f"Invalid model name. Choose from: {', '.join(valid_models)}")
        return True

    def parse_args(self, args: Optional[list] = None) -> Dict[str, Any]:
        parsed_args = self.parser.parse_args(args)
        
        # Validate input file
        self.validate_input(parsed_args.input_file)
        
        # Validate model selection
        self.validate_model(parsed_args.model)
        
        return vars(parsed_args)

def main():
    try:
        cli = CLIHandler()
        args = cli.parse_args()
        
        # TODO: Initialize and run the processing pipeline
        print(f"Processing file: {args['input_file']}")
        print(f"Using model: {args['model']}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()