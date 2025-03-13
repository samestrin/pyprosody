from dataclasses import dataclass
from typing import List, Dict, Set
import re
from collections import Counter
from ..text_processing.segmentation import TextSegment

@dataclass
class DiscourseFeatures:
    discourse_markers: Dict[str, int]
    emphasis_patterns: List[str]
    rhetorical_devices: List[str]
    formality_score: float
    repetition_patterns: List[str]

@dataclass
class PragmaticScore:
    features: DiscourseFeatures
    emotional_intensity: float
    certainty_level: float
    formality_level: float

class PragmaticAnalyzer:
    def __init__(self):
        self.discourse_markers = {
            'causal': {'because', 'therefore', 'thus', 'hence', 'so'},
            'contrast': {'however', 'but', 'although', 'nevertheless', 'yet'},
            'emphasis': {'indeed', 'certainly', 'clearly', 'obviously', 'notably'},
            'sequence': {'first', 'then', 'finally', 'next', 'subsequently'},
            'elaboration': {'specifically', 'particularly', 'especially', 'namely'}
        }
        
        self.formal_indicators = {
            'moreover', 'furthermore', 'consequently', 'nevertheless', 'regarding',
            'concerning', 'whereas', 'hereby', 'therein', 'thereafter'
        }
        
        self.informal_indicators = {
            'like', 'well', 'you know', 'kind of', 'sort of', 'basically',
            'actually', 'pretty much', 'stuff', 'things'
        }
        
    def analyze_segment(self, segment: TextSegment) -> PragmaticScore:
        text = segment.text
        
        # Analyze discourse markers
        markers = self._identify_discourse_markers(text)
        
        # Identify emphasis patterns
        emphasis = self._identify_emphasis_patterns(text)
        
        # Detect rhetorical devices
        rhetorical = self._identify_rhetorical_devices(text)
        
        # Calculate formality score
        formality = self._calculate_formality(text)
        
        # Find repetition patterns
        repetition = self._identify_repetition(text)
        
        # Calculate emotional intensity based on features
        emotional_intensity = self._calculate_emotional_intensity(
            markers, emphasis, rhetorical, repetition
        )
        
        # Calculate certainty level
        certainty = self._calculate_certainty_level(markers, text)
        
        return PragmaticScore(
            features=DiscourseFeatures(
                discourse_markers=markers,
                emphasis_patterns=emphasis,
                rhetorical_devices=rhetorical,
                formality_score=formality,
                repetition_patterns=repetition
            ),
            emotional_intensity=emotional_intensity,
            certainty_level=certainty,
            formality_level=formality
        )
    
    def _identify_discourse_markers(self, text: str) -> Dict[str, int]:
        text_lower = text.lower()
        markers = {}
        
        for category, words in self.discourse_markers.items():
            count = sum(1 for word in words if word in text_lower)
            if count > 0:
                markers[category] = count
                
        return markers
    
    def _identify_emphasis_patterns(self, text: str) -> List[str]:
        patterns = []
        
        # Check for capitalization emphasis
        if re.search(r'\b[A-Z]{2,}\b', text):
            patterns.append('capitalization')
            
        # Check for repetitive punctuation
        if re.search(r'[!?]{2,}', text):
            patterns.append('multiple_punctuation')
            
        # Check for italics markers
        if re.search(r'[*_].+?[*_]', text):
            patterns.append('italics_markers')
            
        return patterns
    
    def _identify_rhetorical_devices(self, text: str) -> List[str]:
        devices = []
        
        # Check for rhetorical questions
        if re.search(r'\b(?:why|how|what|when|where|who)\b.*\?', text.lower()):
            devices.append('rhetorical_question')
            
        # Check for parallel structures
        if re.search(r'(\b\w+\b).*\1', text):
            devices.append('parallel_structure')
            
        # Check for comparative structures
        if any(word in text.lower() for word in ['like', 'as', 'than']):
            devices.append('comparison')
            
        return devices
    
    def _calculate_formality(self, text: str) -> float:
        text_lower = text.lower()
        formal_count = sum(1 for word in self.formal_indicators if word in text_lower)
        informal_count = sum(1 for word in self.informal_indicators if word in text_lower)
        
        if formal_count + informal_count == 0:
            return 0.5
        
        return formal_count / (formal_count + informal_count)
    
    def _identify_repetition(self, text: str) -> List[str]:
        words = text.lower().split()
        word_counts = Counter(words)
        
        return [word for word, count in word_counts.items() 
                if count > 1 and word not in {'the', 'a', 'an', 'and', 'or', 'but'}]
    
    def _calculate_emotional_intensity(self, 
                                    markers: Dict[str, int],
                                    emphasis: List[str],
                                    rhetorical: List[str],
                                    repetition: List[str]) -> float:
        score = 0.0
        
        # Weight discourse markers
        score += sum(markers.values()) * 0.1
        
        # Weight emphasis patterns
        score += len(emphasis) * 0.2
        
        # Weight rhetorical devices
        score += len(rhetorical) * 0.15
        
        # Weight repetition
        score += len(repetition) * 0.1
        
        return min(1.0, score)
    
    def _calculate_certainty_level(self, 
                                 markers: Dict[str, int], 
                                 text: str) -> float:
        certainty_indicators = {
            'certainly', 'definitely', 'surely', 'clearly',
            'undoubtedly', 'absolutely', 'obviously'
        }
        uncertainty_indicators = {
            'maybe', 'perhaps', 'possibly', 'probably',
            'might', 'could', 'may', 'seems'
        }
        
        text_lower = text.lower()
        certainty_count = sum(1 for word in certainty_indicators if word in text_lower)
        uncertainty_count = sum(1 for word in uncertainty_indicators if word in text_lower)
        
        # Add emphasis markers as certainty indicators
        certainty_count += markers.get('emphasis', 0)
        
        if certainty_count + uncertainty_count == 0:
            return 0.5
            
        return certainty_count / (certainty_count + uncertainty_count)