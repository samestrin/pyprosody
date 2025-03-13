from dataclasses import dataclass
from typing import List, Optional
import spacy
from datetime import datetime

@dataclass
class TextSegment:
    id: str
    text: str
    segment_type: str  # 'paragraph', 'sentence', or 'phrase'
    start_pos: int
    end_pos: int
    parent_id: Optional[str] = None

class TextSegmenter:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def segment_text(self, text: str) -> List[TextSegment]:
        segments = []
        doc = self.nlp(text)
        
        # Process paragraphs (split by double newlines)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for p_idx, para in enumerate(paragraphs):
            para_id = f'p{p_idx}'
            para_start = text.find(para)
            para_end = para_start + len(para)
            
            segments.append(TextSegment(
                id=para_id,
                text=para,
                segment_type='paragraph',
                start_pos=para_start,
                end_pos=para_end
            ))
            
            # Process sentences within paragraph
            para_doc = self.nlp(para)
            for s_idx, sent in enumerate(para_doc.sents):
                sent_id = f'{para_id}_s{s_idx}'
                segments.append(TextSegment(
                    id=sent_id,
                    text=sent.text,
                    segment_type='sentence',
                    start_pos=para_start + sent.start_char,
                    end_pos=para_start + sent.end_char,
                    parent_id=para_id
                ))
                
                # Process phrases within sentence
                for p_idx, phrase in enumerate(self._extract_phrases(sent)):
                    phrase_id = f'{sent_id}_ph{p_idx}'
                    segments.append(TextSegment(
                        id=phrase_id,
                        text=phrase.text,
                        segment_type='phrase',
                        start_pos=para_start + phrase.start_char,
                        end_pos=para_start + phrase.end_char,
                        parent_id=sent_id
                    ))
        
        return segments

    def _extract_phrases(self, sent):
        """Extract meaningful phrases from a sentence using syntactic parsing"""
        phrases = []
        for chunk in sent.noun_chunks:
            phrases.append(chunk)
        for token in sent:
            if token.dep_ in ['ROOT', 'VERB']:
                phrases.append(token)
        return sorted(phrases, key=lambda x: x.start_char)