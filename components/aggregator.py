"""
Author: Jaskaran Singh Kawatra
Aggregating findings into a final diagnosis
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

class DiagnosisAggregator:
    """
    Aggregates multiple findings (and their sentiment) into a final diagnosis using rule-based logic.
    """
    
    def __init__(self, group_col: str = "entity_id", delimiter: str = "; "):
        """
        Args:
            group_col (str): Column name by which to group rows (e.g., 'entity_id').
            delimiter (str): Delimiter to join multiple diagnoses in a single group.
        """
        self.group_col = group_col
        self.delimiter = delimiter
    
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates diagnoses in each group, prioritizing positive findings if available.
        
        Args:
            df (pd.DataFrame): The exploded DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with a new column 'final_diagnosis' for each group.
        """
        if self.group_col not in df.columns:
            raise ValueError(f"Column '{self.group_col}' not found in DataFrame.")
        
        def _aggregate_diagnosis(group: pd.DataFrame) -> pd.DataFrame:
            positives = group.loc[group["sentiment_prediction"] == "Positive", "finding"].tolist()
            negatives = group.loc[group["sentiment_prediction"] == "Negative", "finding"].tolist()
            
            if positives:
                chosen = positives
            else:
                chosen = negatives
                
            final_str = self.delimiter.join(chosen) if chosen else ""
            group["final_diagnosis"] = final_str
            return group
        
        df = df.groupby(self.group_col, group_keys=False).apply(_aggregate_diagnosis)
        return df