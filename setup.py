from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyprosody",
    version="0.1.0",
    author="Sam Estrin",    
    description="Read stories with prosody using AI models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samestrin/pyprosody",        
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'nltk>=3.6.7',
        'transformers>=4.12.5',
        'pydub>=0.25.1',
        'soundfile>=0.10.3',
        'tqdm>=4.62.3',
        'torch>=2.6.0',
        'torchaudio>=2.6.0',
        'TTS>=0.22.0',
        'spacy>=3.8.4',
        'scipy>=1.15.2',
        'scikit-learn>=1.6.1',
        'pandas>=1.5.3',
        'matplotlib>=3.10.1',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.910',
        ],
    },
    entry_points={
        'console_scripts': [
            'pyprosody=pyprosody.cli.main:main',
        ],
    },
    python_requires='>=3.8',
    include_package_data=True,
)