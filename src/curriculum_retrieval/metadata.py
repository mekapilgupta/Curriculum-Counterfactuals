"""
Zero-Shot Query Metadata Predictor.
Predicts educational curriculum subject and topic strictly from the raw question text alone,
ensuring zero gold-label leakage at inference time.
"""

from typing import Dict, List, Optional
import numpy as np


class ZeroShotMetadataPredictor:
    """Predicts subject and topic categories from question text using vocabulary/keyword semantic matching."""

    SUBJECT_KEYWORDS = {
        "natural science": [
            "animal", "plant", "cell", "energy", "force", "magnet", "motion", "gravity", "mass",
            "ecosystem", "organism", "rock", "mineral", "earth", "sun", "moon", "planet", "water",
            "temperature", "heat", "chemical", "atom", "molecule", "species", "habitat", "weather",
            "climate", "electric", "circuit", "light", "sound", "wave", "matter", "solid", "liquid", "gas"
        ],
        "social science": [
            "government", "law", "president", "citizen", "country", "state", "war", "history",
            "map", "continent", "ocean", "economy", "money", "trade", "tax", "vote", "election",
            "culture", "society", "colony", "revolution", "constitution", "rights", "native", "settler"
        ],
        "language science": [
            "verb", "noun", "adjective", "sentence", "punctuation", "comma", "period", "capitalize",
            "capitalization", "pronoun", "spelling", "grammar", "paragraph", "syllable", "rhyme",
            "metaphor", "simile", "prefix", "suffix", "tense", "plural", "singular", "word", "meaning"
        ],
    }

    TOPIC_KEYWORDS = {
        "biology": ["cell", "organism", "animal", "plant", "ecosystem", "species", "body", "dna", "gene"],
        "physics": ["force", "motion", "magnet", "gravity", "speed", "energy", "circuit", "electric", "light", "sound"],
        "earth-science": ["rock", "mineral", "earth", "weather", "volcano", "earthquake", "ocean", "fossil", "plate"],
        "civics": ["government", "law", "citizen", "rights", "vote", "president", "constitution", "court"],
        "history": ["war", "colony", "revolution", "century", "ancient", "settler", "president", "history"],
        "economics": ["money", "economy", "trade", "tax", "cost", "producer", "consumer", "good", "service"],
        "verbs": ["verb", "past tense", "action", "tense", "irregular"],
        "capitalization": ["capitalize", "capital letter", "proper noun", "title"],
        "punctuation": ["comma", "period", "apostrophe", "quotation", "exclamation", "question mark"],
        "vocabulary": ["meaning", "definition", "synonym", "antonym", "context clue"],
    }

    def predict_subject(self, question_text: str) -> str:
        q_lower = question_text.lower()
        scores = {}
        for subject, kws in self.SUBJECT_KEYWORDS.items():
            count = sum(1 for kw in kws if kw in q_lower)
            scores[subject] = count
        best_subject = max(scores.items(), key=lambda x: x[1])
        return best_subject[0] if best_subject[1] > 0 else "natural science"

    def predict_topic(self, question_text: str) -> Optional[str]:
        q_lower = question_text.lower()
        scores = {}
        for topic, kws in self.TOPIC_KEYWORDS.items():
            count = sum(1 for kw in kws if kw in q_lower)
            scores[topic] = count
        best_topic = max(scores.items(), key=lambda x: x[1])
        return best_topic[0] if best_topic[1] > 0 else None
