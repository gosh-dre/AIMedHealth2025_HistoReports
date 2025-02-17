"""
Author: Jaskaran Singh Kawatra
Relevance Prediction with SetFit
"""
import os
import glob
import logging
from typing import List
from setfit import SetFitModel

logger = logging.getLogger(__name__)



class RelevancePredictor:
    """
    Handles loading a Relevance (SetFit) model and applying it to text data to predict
    whether each input is relevant or not.
    """
    
    def __init__(self, model_dir: str):
        """
        Initializes the RelevancePredictor with a SetFit model from a directory.
        
        Args:
            model_dir (str): Path to the directory containing the trained relevance model.
        """
        logger.info("Loading relevance model...")
        possible_dirs = glob.glob(os.path.join(model_dir, "*/"))
        if not possible_dirs:
            raise ValueError(f"No checkpoints found in {model_dir}.")
        latest_checkpoint = max(possible_dirs, key=os.path.getmtime)
        logger.info(f"Using relevance model from checkpoint: {latest_checkpoint}")
        
        self.model = SetFitModel.from_pretrained(model_dir)
        
    def predict_relevance(self, texts: List[str]) -> List[str]:
        """
        Predict relevance labels for the given list of texts.
        
        Args:
            texts (List[str]): List of input texts to classify.
        
        Returns:
            List[str]: List of predicted labels: "Relevant" or "Not Relevant".
        """
        predictions = self.model.predict(texts)
        # Convert booleans to int (True -> 1, False -> 0)
        int_preds = []
        for p in predictions:
            val = int(p)
            int_preds.append(val)
        
        label_map = {1: "Relevant", 0: "Not Relevant"}
        return [label_map[val] for val in int_preds]
    