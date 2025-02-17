"""
Author: Jaskaran Singh Kawatra
Command-line entry point for the pipeline
"""
 
import argparse
import logging
from .pipeline import InferencePipeline
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
def main():
    """
    Main entry point for command line execution.
    """
    parser = argparse.ArgumentParser(description="Run Inference Pipeline")
    parser.add_argument("--input_csv", type=str, default="infer.csv", help="Path to input CSV")
    parser.add_argument("--output_csv", type=str, default="infer_results.csv", help="Path to output CSV")
    parser.add_argument("--relevance_model_dir", type=str, required=True, help="Path to relevance model folder")
    parser.add_argument("--ner_model_dir", type=str, required=True, help="Path to NER model folder")
    parser.add_argument("--sentiment_model_dir", type=str, required=True, help="Path to sentiment model folder")
    parser.add_argument("--tinyllama_model_dir", type=str, required=True, help="Path to TinyLlama model folder")
    parser.add_argument("--nli_model_dir", type=str, required=True, help="Path to NLI model folder")
    parser.add_argument("--deepseek_model_dir", type=str, required=True, help="Path to Deepseek model folder")
 
    args = parser.parse_args()
 
    pipeline = InferencePipeline(
        relevance_model_dir=args.relevance_model_dir,
        ner_model_dir=args.ner_model_dir,
        sentiment_model_dir=args.sentiment_model_dir,
        tinyllama_model_dir=args.tinyllama_model_dir,
        nli_model_dir=args.nli_model_dir,
        deepseek_model_dir=args.deepseek_model_dir
    )
 
    pipeline.run_pipeline(input_csv=args.input_csv, output_csv=args.output_csv)
 
if __name__ == "__main__":
    main()