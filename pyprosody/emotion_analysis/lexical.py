from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from afinn import Afinn
import nltk
from nltk.corpus import sentiwordnet as swn
from ..text_processing.segmentation import TextSegment

@dataclass
class LexicalScore:
    afinn_score: float
    positive_score: float
    negative_score: float
    objective_score: float
    compound_score: float

class LexicalAnalyzer:
    def __init__(self):
        # Initialize AFINN
        self.afinn = Afinn()
        
        # Download required NLTK data
        nltk.download('sentiwordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        
    def analyze_segment(self, segment: TextSegment) -> LexicalScore:
        text = segment.text
        
        # Get AFINN score
        afinn_score = self.afinn.score(text)
        
        # Get SentiWordNet scores
        pos_score, neg_score, obj_score = self._get_sentiwordnet_scores(text)
        
        # Calculate compound score (weighted average)
        compound_score = (afinn_score + (pos_score - neg_score)) / 2
        
        return LexicalScore(
            afinn_score=afinn_score,
            positive_score=pos_score,
            negative_score=neg_score,
            objective_score=obj_score,
            compound_score=compound_score
        )
    
    def _get_sentiwordnet_scores(self, text: str) -> tuple[float, float, float]:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        
        pos_score = 0.0
        neg_score = 0.0
        obj_score = 0.0
        count = 0
        
        for word, tag in tagged:
            # Convert POS tag to WordNet format
            pos = self._get_wordnet_pos(tag)
            if pos:
                # Get SentiWordNet synsets
                synsets = list(swn.senti_synsets(word, pos))
                if synsets:
                    # Average scores for all synsets
                    synset = synsets[0]  # Use first synset
                    pos_score += synset.pos_score()
                    neg_score += synset.neg_score()
                    obj_score += synset.obj_score()
                    count += 1
        
        if count > 0:
            return (pos_score/count, neg_score/count, obj_score/count)
        return (0.0, 0.0, 1.0)  # Default to neutral if no words found
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        if treebank_tag.startswith('J'):
            return 'a'  # Adjective
        elif treebank_tag.startswith('V'):
            return 'v'  # Verb
        elif treebank_tag.startswith('N'):
            return 'n'  # Noun
        elif treebank_tag.startswith('R'):
            return 'r'  # Adverb
        return ''