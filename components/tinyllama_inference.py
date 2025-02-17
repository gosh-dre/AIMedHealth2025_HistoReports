"""
Author: Jaskaran Singh Kawatra
Text generation with TinyLlama
"""

import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline

logger = logging.getLogger(__name__)

class TinyLlamaInference:
    """
    Encapsulates text generation (inference) logic using a TinyLlama model
    """
    
    def __init__(self, model_dir: str):
        """
        Loads the tokenizer and model for TinyLlama text generation.
        
        Args:
            model_dir (str): Path to the directory containing the TinyLlama model
        """
        logger.info("Loading TinyLlama pipeline...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir)
        
        self.pipeline = hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=256
        )
    
    def generate_output(self, final_diagnosis: str, original_text: str) -> str:
        """
        Generate output from TinyLlama given a final diagnosis and the original text.
        
        Args:
            final_diagnosis (str): The aggregated diagnosis.
            original_text (str): The original text string.
        
        Returns:
            str: The generated output from the TinyLlama model.
        """
        message = f"Original Text: {original_text}"
        
        # Build prompt as a conversation-like structure
        messages = [
            {
                "role": "system",
                "content": f"Question: What is the diagnosis?: Context: {message}"
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
            max_new_tokens=256,
            eos_token_id=eos_token_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.5
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