"""Example usage of the Prompt Evolution algorithm with different LLM backends."""
import os
from dotenv import load_dotenv
import openai
from prompt_evolution import PromptEvolution, EvolutionConfig

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def evaluate_prompt(prompt: str) -> float:
    """Evaluate the quality of a prompt using OpenAI's API.
    
    This is just an example evaluation function. You should modify this
    based on your specific requirements and evaluation criteria.
    
    Args:
        prompt: The prompt to evaluate
        
    Returns:
        A fitness score between 0 and 1
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # This is a simple example - you should implement your own
        # evaluation criteria based on your needs
        response_length = len(response.choices[0].message.content)
        return min(response_length / 500, 1.0)  # Normalize to [0,1]
        
    except Exception as e:
        print(f"Error evaluating prompt: {e}")
        return 0.0

def evolve_with_model(base_prompt: str, llm_type: str, model_name: str = None) -> None:
    """Run evolution with specified LLM model."""
    print(f"\nRunning evolution with {llm_type} model: {model_name or 'default'}")
    print("-" * 80)
    
    # Configure the evolution parameters
    config = EvolutionConfig(
        population_size=20,
        num_generations=10,
        mutation_rate=0.3,
        crossover_rate=0.7,
        elite_size=2,
        tournament_size=3,
        max_prompt_length=500,
        min_prompt_length=50,
        patience=3,
        llm_temperature=0.7,
        max_tokens=100
    )
    
    # Create and run the evolution
    evolution = PromptEvolution(
        config=config,
        llm_type=llm_type,
        model_name=model_name
    )
    
    evolution.initialize_population(base_prompt)
    best_prompt = evolution.evolve(evaluate_prompt)
    
    print(f"\nBest evolved prompt (fitness: {best_prompt.fitness:.4f}):")
    print(best_prompt.text)

def main():
    # Initial prompt to evolve
    base_prompt = """Create a comprehensive analysis of the current market trends 
    in artificial intelligence, focusing on practical applications in business."""
    
    print(f"Original prompt:\n{base_prompt}\n")
    
    # Example with OpenAI GPT-3.5
    evolve_with_model(base_prompt, "openai", "gpt-3.5-turbo")
    
    # Example with OpenAI GPT-4
    evolve_with_model(base_prompt, "openai", "gpt-4")
    
    # Example with HuggingFace model
    evolve_with_model(base_prompt, "huggingface", "facebook/opt-350m")
    
    # Example with a different HuggingFace model
    evolve_with_model(base_prompt, "huggingface", "google/flan-t5-base")

if __name__ == "__main__":
    main()
