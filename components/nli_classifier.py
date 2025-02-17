"""
Author: Jaskaran Singh Kawatra
Natural Language Inference (NLI) Classification
"""
 
import logging
from typing import Dict, Any
from transformers import pipeline as hf_pipeline
 
logger = logging.getLogger(__name__)
 
class NLIClassifier:
    """
    Handles Natural Language Inference classification on a pair of texts.
    """
 
    def __init__(self, model_dir: str):
        """
        Loads a text classification pipeline for NLI.
 
        Args:
            model_dir (str): Path to the directory containing the NLI model.
        """
        logger.info(f"Loading NLI model from {model_dir}...")
        self.nli_pipe = hf_pipeline("text-classification", model=model_dir)
 
    def classify(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Classify the relationship between premise and hypothesis.
 
        Args:
            premise (str): The text to consider as premise.
            hypothesis (str): The text to consider as hypothesis.
 
        Returns:
            Dict[str, Any]: A dict with 'label' and 'score', e.g., 
                {"label": "ENTAILMENT", "score": 0.97}
        """
        results = self.nli_pipe([{"text": premise, "text_pair": hypothesis}])
        return results[0]  # top result