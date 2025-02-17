"""
Author: Jaskaran Singh Kawatra
Sentiment Analysis (Positive/Negative) with SetFit
"""

import logging
from typing import List
from setfit import SetFitModel

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Handes sentiment classification (positive vs negative) for extracted findings using a SetFit model.
    """
    
    def __init__(self, model_dir: str):
        """
        Initializesthe SentimentAnalyzer.
        
        Args:
            model_dir (str): Path to the directory containing the sentiment model.
        """
        logger.info(f"Loading sentiment model from {model_dir}...")
        self.model = SetFitModel.from_pretrained(model_dir)
        
    def predict_sentiment(self, texts: List[str]) -> List[str]:
        """
        Predict sentiment for a list of texts.
        
        Args:
            texts (List[str]): A list of extracted findings.
        
        Returns:
            List[str]: List of predicted sentiment labels. e.g, "Positive" or "Negative".
        """
        predictions = self.model.predict(texts)
        return predictions