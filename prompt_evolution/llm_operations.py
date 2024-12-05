"""LLM-based genetic operations for prompt evolution."""
from abc import ABC, abstractmethod
from typing import Optional, List
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

from .config import EvolutionConfig


class LLMOperator(ABC):
    """Abstract base class for LLM operations."""
    
    def __init__(self, config: EvolutionConfig):
        """Initialize LLM operator with configuration."""
        self.config = config
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate text using the LLM."""
        pass


class OpenAIOperator(LLMOperator):
    """OpenAI GPT-based operator."""
    
    def __init__(self, config: EvolutionConfig, model_name: str = "gpt-3.5-turbo"):
        """Initialize OpenAI operator.
        
        Args:
            config: Evolution configuration
            model_name: Name of the OpenAI model to use
        """
        super().__init__(config)
        self.model_name = model_name
    
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate text using OpenAI's API."""
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens
        )
        return response.choices[0].message.content


class HuggingFaceOperator(LLMOperator):
    """HuggingFace model-based operator."""
    
    def __init__(self, config: EvolutionConfig, model_name: str = "facebook/opt-350m"):
        """Initialize HuggingFace operator.
        
        Args:
            config: Evolution configuration
            model_name: Name of the HuggingFace model to use
        """
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def generate(self, prompt: str, system_prompt: str) -> str:
        """Generate text using HuggingFace model."""
        full_prompt = f"{system_prompt}\n\nInput: {prompt}\n\nOutput:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.llm_temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(full_prompt):].strip()


class GeneticOperations:
    """LLM-based genetic operations for prompt evolution."""
    
    def __init__(self, llm_operator: LLMOperator):
        """Initialize genetic operations with specified LLM operator."""
        self.llm = llm_operator
    
    def mutate(self, text: str) -> str:
        """Apply LLM-based mutation to the prompt."""
        system_prompt = """You are helping to mutate a prompt. Apply ONE of these changes:
1. Replace words with more precise or contextually appropriate synonyms
2. Rephrase while preserving the core meaning
3. Add relevant context or specifications
4. Remove redundant information
5. Improve clarity and conciseness

Ensure the output:
- Maintains grammatical correctness
- Preserves the original intent
- Makes meaningful improvements

Return ONLY the mutated prompt, no explanations."""
        
        return self.llm.generate(text, system_prompt)
    
    def crossover(self, parent1: str, parent2: str) -> str:
        """Perform LLM-based crossover between two parent prompts."""
        system_prompt = """You are helping to combine two prompts into a new one.
Create a new prompt that:
1. Combines the best elements from both parents
2. Maintains coherence and fluency
3. Preserves important information from both sources
4. Creates a meaningful synthesis

Return ONLY the combined prompt, no explanations."""
        
        prompt = f"Parent 1: {parent1}\nParent 2: {parent2}"
        return self.llm.generate(prompt, system_prompt)
    
    def insert_words(self, text: str) -> str:
        """Insert contextually appropriate words or phrases."""
        system_prompt = """You are helping to enhance a prompt by inserting relevant words or phrases.
Add content that:
1. Provides additional context or specificity
2. Enhances clarity or precision
3. Maintains natural flow
4. Adds meaningful information

Return ONLY the modified prompt, no explanations."""
        
        return self.llm.generate(text, system_prompt)
    
    def swap_words(self, text: str) -> str:
        """Swap words or phrases to improve prompt structure."""
        system_prompt = """You are helping to improve a prompt by rearranging its components.
Modify the prompt by:
1. Reordering elements for better flow
2. Improving emphasis through word placement
3. Enhancing readability
4. Maintaining coherence

Return ONLY the rearranged prompt, no explanations."""
        
        return self.llm.generate(text, system_prompt)
    
    def substitute_words(self, text: str) -> str:
        """Replace words with more effective alternatives."""
        system_prompt = """You are helping to improve a prompt by substituting words.
Replace words to:
1. Use more precise or appropriate terminology
2. Enhance clarity and impact
3. Maintain or improve meaning
4. Use domain-specific language where appropriate

Return ONLY the modified prompt, no explanations."""
        
        return self.llm.generate(text, system_prompt)
    
    def delete_words(self, text: str) -> str:
        """Remove unnecessary words or phrases."""
        system_prompt = """You are helping to improve a prompt by removing unnecessary elements.
Modify the prompt by:
1. Removing redundant information
2. Eliminating unnecessary qualifiers
3. Making the prompt more concise
4. Preserving all essential information

Return ONLY the streamlined prompt, no explanations."""
        
        return self.llm.generate(text, system_prompt)
