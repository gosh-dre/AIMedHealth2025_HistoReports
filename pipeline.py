"""
Author: Jaskaran Singh Kawatra
End-to-end pipeline orchestration
"""
 
import logging
import numpy as np
import pandas as pd
from typing import Union
 
# Import the components (relative import)
from .components.relevance_predictor import RelevancePredictor
from .components.ner_extractor import NERExtractor
from .components.sentiment_analyzer import SentimentAnalyzer
from .components.aggregator import DiagnosisAggregator
from .components.tinyllama_inference import TinyLlamaInference
from .components.deepseek_inference import DeepSeekInference
from .components.nli_classifier import NLIClassifier
 
logger = logging.getLogger(__name__)
 
class InferencePipeline:
    """
    Orchestrates the entire end-to-end inference flow.
    """
 
    def __init__(
        self,
        relevance_model_dir: str,
        ner_model_dir: str,
        sentiment_model_dir: str,
        tinyllama_model_dir: str,
        nli_model_dir: str,
        deepseek_model_dir: str
    ):
        """
        Initializes all components required for the pipeline.
        """
        # 1. Relevancy
        self.relevance_predictor = RelevancePredictor(relevance_model_dir)
 
        # 2. NER
        self.ner_extractor = NERExtractor(ner_model_dir)
 
        # 3. Sentiment
        self.sentiment_analyzer = SentimentAnalyzer(sentiment_model_dir)
 
        # 4. Aggregator
        self.diagnosis_aggregator = DiagnosisAggregator(group_col="entity_id", delimiter="; ")
 
        # 5. TinyLlama Inference
        self.tinyllama_inference = TinyLlamaInference(tinyllama_model_dir)
 
        # 6. DeepSeek Inference
        self.deepseek_inference = DeepSeekInference(deepseek_model_dir)
 
        # 7. NLI
        self.nli_classifier = NLIClassifier(nli_model_dir)
 
    def run_pipeline(self, input_csv: str, output_csv: str) -> None:
        """
        Execute the pipeline steps and save the final DataFrame to CSV.
 
        1. Load data.
        2. Predict relevancy.
        3. Run NER for relevant rows.
        4. Explode findings.
        5. Run sentiment classification on each finding.
        6. Aggregate final diagnosis.
        7. Generate text with TinyLlama & run NLI.
        8. Generate text with DeepSeek & run NLI.
        9. Save results to CSV.
        """
        # Step 1: Load data
        df = pd.read_csv(input_csv)
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain a column named 'text'.")
 
        # Initialize columns if they don't exist
        if "relevance_prediction" not in df.columns:
            df["relevance_prediction"] = ""
        if "extracted_findings" not in df.columns:
            df["extracted_findings"] = ""
 
        # Step 2: Predict relevancy
        logger.info("Running Relevance Prediction...")
        df["relevance_prediction"] = self.relevance_predictor.predict_relevance(df["text"].tolist())
 
        # Step 3: Run NER for relevant rows
        logger.info("Running NER on relevant texts...")
        df = self._run_ner_and_store_findings(df)
 
        # Step 4: Explode findings
        df_exploded = self._explode_findings_for_sentiment(df)
 
        # Step 5: Sentiment classification
        df_exploded = self._run_sentiment(df_exploded)
 
        # Step 6: Aggregate final diagnosis
        df_exploded = self.diagnosis_aggregator.aggregate(df_exploded)
 
        # Step 7: TinyLlama Inference + NLI
        logger.info("Running TinyLlama inference + NLI...")
        df_exploded["tinyllama_output"] = df_exploded.apply(
            lambda row: self.tinyllama_inference.generate_output(
                str(row["final_diagnosis"]),
                str(row["text"])
            ),
            axis=1
        )
        nli_results_tiny = df_exploded.apply(
            lambda row: self.nli_classifier.classify(
                premise=str(row["tinyllama_output"]),
                hypothesis=str(row["final_diagnosis"])
            ),
            axis=1
        )
        df_exploded["tinyllama_nli_label"] = nli_results_tiny.apply(lambda x: x["label"])
        df_exploded["tinyllama_nli_score"] = nli_results_tiny.apply(lambda x: x["score"])
 
        # Step 8: DeepSeek Inference + NLI
        logger.info("Running DeepSeek inference + NLI...")
        df_exploded["deepseek_output"] = df_exploded.apply(
            lambda row: self.deepseek_inference.generate_output(
                str(row["text"])
            ),
            axis=1
        )
 
        # Prepare premise for deepseek's NLI (strip out <think> blocks if present)
        def extract_post_think(text: str) -> str:
            """
            Extract the text after '</think>' if present, else return original.
            """
            think_end_pos = text.find('</think')
            if think_end_pos != -1:
                return text[think_end_pos + 8:].strip()
            return text
 
        nli_results_deepseek = df_exploded.apply(
            lambda row: self.nli_classifier.classify(
                premise=extract_post_think(str(row["deepseek_output"])),
                hypothesis=str(row["final_diagnosis"])
            ),
            axis=1
        )
        df_exploded["deepseek_nli_label"] = nli_results_deepseek.apply(lambda x: x["label"])
        df_exploded["deepseek_nli_score"] = nli_results_deepseek.apply(lambda x: x["score"])
 
        # Step 9: Save to CSV
        df_exploded.to_csv(output_csv, index=False)
        logger.info(f"Final pipeline output saved to {output_csv}")
 
    def _run_ner_and_store_findings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each row in df that is 'Relevant', call the GLiNER model
        to get a list of findings. Then store them in a semicolon-delimited
        string in 'extracted_findings'.
 
        Args:
            df (pd.DataFrame): Dataframe with 'relevance_prediction' and 'text' columns.
 
        Returns:
            pd.DataFrame: Updated dataframe with 'extracted_findings' column populated.
        """
        all_findings = []
        for _, row in df.iterrows():
            if row["relevance_prediction"] == "Relevant":
                text_val = str(row["text"])
                findings_list = self.ner_extractor.extract_findings(text_val)
                joined_findings = "; ".join(findings_list)
                all_findings.append(joined_findings)
            else:
                all_findings.append("")
        df["extracted_findings"] = all_findings
        return df
 
    @staticmethod
    def _explode_findings_for_sentiment(df: pd.DataFrame) -> pd.DataFrame:
        """
        Take df with a semicolon-delimited 'extracted_findings' column
        and explode it so each row has exactly 1 finding.
 
        Args:
            df (pd.DataFrame): Original DataFrame with 'extracted_findings'.
 
        Returns:
            pd.DataFrame: Exploded DataFrame with 'finding' column.
        """
        df["findings_list"] = df["extracted_findings"].apply(
            lambda x: [f.strip() for f in x.split(";")] if x else []
        )
        df_exploded = df.explode("findings_list", ignore_index=True)
        df_exploded.rename(columns={"findings_list": "finding"}, inplace=True)
        return df_exploded
 
    def _run_sentiment(self, df_exploded: pd.DataFrame) -> pd.DataFrame:
        """
        Performs sentiment classification on rows marked "Relevant" with non-empty findings.
 
        Args:
            df_exploded (pd.DataFrame): Exploded DataFrame.
 
        Returns:
            pd.DataFrame: Updated DataFrame with 'sentiment_prediction' column.
        """
        # 1. Initialize an empty column "sentiment_prediction" for all rows
        df_exploded["sentiment_prediction"] = np.nan
 
        # 2. Mask relevant rows that have a non-empty finding
        mask_relevant = (
            (df_exploded["relevance_prediction"] == "Relevant") &
            (df_exploded["finding"].notna()) &
            (df_exploded["finding"] != "")
        )
 
        # 3. Extract only those findings
        relevant_findings = df_exploded.loc[mask_relevant, "finding"].tolist()
 
        # 4. Run the sentiment model on those findings
        sentiments = self.sentiment_analyzer.predict_sentiment(relevant_findings)
 
        # 5. Assign results back to the DataFrame
        df_exploded.loc[mask_relevant, "sentiment_prediction"] = sentiments
 
        return df_exploded