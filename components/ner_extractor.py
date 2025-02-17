"""
Author: Jaskaran Singh Kawatra
Named Entity Recognition with GLiNER
"""

import logging
from typing import List
from gliner import GLiNER

logger = logging.getLogger(__name__)

class NERExtractor:
    """
    Performs Named Entity Recognition using GLiNER on relevant texts.
    """
    
    def __init__(self, model_dir: str, threshold: float = 0.5):
        """
        Initializes the NER extractor with a GLiNER model.
        
        Args:
            model_dir (str): Path to the directory containing the NER model.
            threshold (float): Temperature for entity prediction --- Revisit
        """
        logger.info(f"Loading NER model from {model_dir}...")
        self.model = GLiNER.from_pretrained(model_dir)
        self.threshold = threshold
        # The GLiNER model usage requires a context and labels. We define them here for convenience:
        self.context = "Please find the diagnoses from the given text string."
        self.labels = ["Diagnosis"]
        
    def extract_findings(self, text: str) -> List[str]:
        """
        Run GLiNER prediction on a single text and extract recognized entities.
        
        Args:
            text (str): Input text.
        
        Returns:
            List[str]: List of extracted entity texts.
        """
        entities = self.model.predict_entities(
            self.context + text,
            self.labels,
            threshold=self.threshold
        )
        return [ent["text"] for ent in entities]