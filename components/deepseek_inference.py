"""
Author: Jaskaran Singh Kawatra
Text generation with DeepSeek
"""
 
import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
 
logger = logging.getLogger(__name__)
 
class DeepSeekInference:
    """
    Encapsulates text generation logic using a DeepSeek model.
    """
 
    def __init__(self, model_dir: str):
        """
        Loads the tokenizer and model for DeepSeek text generation.
 
        Args:
            model_dir (str): Path to the directory containing the DeepSeek model.
        """
        logger.info("Loading DeepSeek pipeline...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
 
        self.pipeline = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256
        )
 
    def generate_output(self, original_text: str) -> str:
        """
        Generate output from DeepSeek given an original text string.
 
        Args:
            original_text (str): The original text string.
 
        Returns:
            str: The generated output from the DeepSeek model.
        """
        messages = [
            {
                "role": "user",
                "content": (
                    f"Question: What is the diagnosis?: "
                    f"Context: The following is a text from a medical report: {original_text}. "
                    "Please initiate your response with '<think>\n'. Please get straight to the point."
                )
            },
            {
                "role": "user",
                "content": "ANSWER"
            },
        ]
 
        prompt = self._build_prompt(messages)
 
        eos_token_id = self.pipeline.tokenizer.eos_token_id
        output = self.pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.7,
            repetition_penalty=1.1,
            pad_token_id=self.pipeline.tokenizer.pad_token_id
        )
 
        generated_text = output[0]["generated_text"]
        return generated_text[len(prompt):]
 
    @staticmethod
    def _build_prompt(messages: List[Dict[str, str]]) -> str:
        """
        Convert the messages (role-based) into a single prompt string.
        """
        prompt = ""
        for msg in messages:
            prompt += f"{msg['role'].upper()}: {msg['content']}\n"
        return prompt