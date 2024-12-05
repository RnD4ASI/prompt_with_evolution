"""Configuration settings for the prompt evolution algorithm."""
from dataclasses import dataclass
from typing import Optional, List, Union, Callable


@dataclass
class EvolutionConfig:
    """Configuration for the prompt evolution process.
    
    Attributes:
        population_size: Number of prompts in each generation
        num_generations: Maximum number of generations to evolve
        elite_size: Number of best prompts to preserve unchanged
        mutation_rate: Probability of mutation (0.0 to 1.0)
        crossover_rate: Probability of crossover (0.0 to 1.0)
        tournament_size: Number of prompts in each tournament selection
        max_prompt_length: Maximum allowed length for prompts
        min_prompt_length: Minimum allowed length for prompts
        num_parents: Number of parents to select for each offspring
        num_offspring: Number of offspring to generate in crossover
        save_history: Whether to save evolution history
        verbose: Whether to print progress information
        fitness_function: Optional custom fitness function
        patience: Generations without improvement before early stopping
        min_fitness_improvement: Minimum improvement to reset patience
        llm_temperature: Temperature for LLM generation (0.0 to 1.0)
        max_tokens: Maximum tokens for LLM generation
    """
    # Population settings
    population_size: int = 50
    num_generations: int = 20
    elite_size: int = 2
    
    # Genetic operator probabilities
    mutation_rate: float = 0.3
    crossover_rate: float = 0.7
    
    # Selection settings
    tournament_size: int = 3
    
    # Prompt constraints
    max_prompt_length: int = 500
    min_prompt_length: int = 50
    
    # Evolution strategy
    num_parents: int = 2
    num_offspring: int = 2
    
    # Output settings
    save_history: bool = True
    verbose: bool = True
    
    # Custom evaluation function
    fitness_function: Optional[Callable] = None
    
    # Early stopping
    patience: int = 5
    min_fitness_improvement: float = 0.01
    
    # LLM settings
    llm_temperature: float = 0.7
    max_tokens: int = 100
    
    def validate(self):
        """Validate configuration parameters."""
        assert 0 <= self.mutation_rate <= 1, "Mutation rate must be between 0 and 1"
        assert 0 <= self.crossover_rate <= 1, "Crossover rate must be between 0 and 1"
        assert self.population_size > 0, "Population size must be positive"
        assert self.num_generations > 0, "Number of generations must be positive"
        assert self.tournament_size > 0, "Tournament size must be positive"
        assert self.tournament_size <= self.population_size, "Tournament size cannot exceed population size"
        assert self.max_prompt_length >= self.min_prompt_length, "Max prompt length must be >= min prompt length"
        assert 0 <= self.llm_temperature <= 1, "LLM temperature must be between 0 and 1"
        assert self.max_tokens > 0, "Max tokens must be positive"
