"""Core implementation of the prompt evolution algorithm.

This module implements a genetic algorithm-based approach to evolve and optimize prompts.
The algorithm follows these key steps:
1. Initialize population with variations of a base prompt
2. For each generation:
   - Evaluate fitness of each prompt
   - Select parents through tournament selection
   - Create offspring through crossover and mutation
   - Apply elitism to preserve best solutions
3. Return the best evolved prompt

The implementation includes various genetic operators:
- Tournament Selection: Randomly samples prompts and selects the fittest
- Single-point Crossover: Combines two parent prompts at a random point
- Multiple Mutation Operations: 
  - Word substitution
  - Word deletion
  - Word insertion
  - Word swapping

Early stopping is implemented to prevent overfitting and reduce computation time.
"""
import random
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from tqdm import tqdm

from .config import EvolutionConfig
from .llm_operations import GeneticOperations, OpenAIOperator, HuggingFaceOperator


@dataclass
class Prompt:
    """Represents a single prompt in the population.
    
    Attributes:
        text: The actual prompt text
        fitness: The evaluated fitness score of the prompt
    """
    text: str
    fitness: float = 0.0
    
    def __lt__(self, other):
        """Enable sorting of prompts based on fitness."""
        return self.fitness < other.fitness


class PromptEvolution:
    """Main class implementing the prompt evolution algorithm.
    
    This class handles the entire evolutionary process including:
    - Population initialization and management
    - Parent selection
    - Offspring generation through genetic operators
    - Population evolution across generations
    - Tracking of best solutions
    """
    
    def __init__(self, config: EvolutionConfig, llm_type: str = "openai", 
                 model_name: Optional[str] = None):
        """Initialize the prompt evolution algorithm.
        
        Args:
            config: Configuration settings for the evolution process
            llm_type: Type of LLM to use ('openai' or 'huggingface')
            model_name: Name of the model to use (optional)
        """
        self.config = config
        self.config.validate()
        self.population: List[Prompt] = []
        self.best_prompt: Optional[Prompt] = None
        self.history = []
        
        # Initialize LLM operator
        if llm_type == "openai":
            model = model_name or "gpt-3.5-turbo"
            llm_operator = OpenAIOperator(config, model)
        elif llm_type == "huggingface":
            model = model_name or "facebook/opt-350m"
            llm_operator = HuggingFaceOperator(config, model)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
        
        self.genetic_ops = GeneticOperations(llm_operator)
        
    def initialize_population(self, base_prompt: str) -> None:
        """Initialize the population with variations of the base prompt.
        
        The population is created by:
        1. Including the original base prompt
        2. Generating variations through mutation operations
        
        Args:
            base_prompt: The initial human-drafted prompt to evolve
        """
        self.population = [Prompt(base_prompt)]
        
        # Generate initial population through mutations
        while len(self.population) < self.config.population_size:
            new_prompt = self._mutate(base_prompt)
            self.population.append(Prompt(new_prompt))
    
    def evolve(self, fitness_function: Callable[[str], float]) -> Prompt:
        """Run the evolution process.
        
        The evolution follows these steps for each generation:
        1. Evaluate fitness of all prompts using provided fitness function
        2. Update best prompt if improvement found
        3. Check early stopping conditions
        4. Generate next generation:
           - Preserve elite prompts
           - Select parents through tournament selection
           - Create offspring through crossover and mutation
        
        Args:
            fitness_function: Function that takes a prompt string and returns
                            a fitness score (higher is better)
            
        Returns:
            The best evolved prompt found during evolution
        """
        best_fitness = float('-inf')
        generations_without_improvement = 0
        
        for generation in tqdm(range(self.config.num_generations)):
            # Evaluate fitness
            for prompt in self.population:
                prompt.fitness = fitness_function(prompt.text)
            
            # Update best prompt
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                self.best_prompt = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Early stopping
            if generations_without_improvement >= self.config.patience:
                if self.config.verbose:
                    print(f"Early stopping at generation {generation}")
                break
            
            # Store history
            if self.config.save_history:
                self.history.append({
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'avg_fitness': np.mean([p.fitness for p in self.population])
                })
            
            # Create next generation
            next_generation = []
            
            # Elitism
            sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            next_generation.extend(sorted_population[:self.config.elite_size])
            
            # Generate offspring
            while len(next_generation) < self.config.population_size:
                if random.random() < self.config.crossover_rate:
                    parents = self._tournament_selection(self.config.num_parents)
                    offspring = self._crossover(parents)
                else:
                    offspring = [self._mutate(random.choice(self.population).text)]
                
                next_generation.extend([Prompt(text) for text in offspring])
            
            self.population = next_generation[:self.config.population_size]
        
        return self.best_prompt
    
    def _tournament_selection(self, num_winners: int) -> List[Prompt]:
        """Select prompts using tournament selection.
        
        Tournament selection works by:
        1. Randomly sampling a subset of prompts (tournament)
        2. Selecting the best prompt from each tournament
        3. Repeating until desired number of parents are selected
        
        This method provides selection pressure while maintaining diversity.
        
        Args:
            num_winners: Number of prompts to select through tournaments
            
        Returns:
            Selected parent prompts for offspring generation
        """
        winners = []
        for _ in range(num_winners):
            tournament = random.sample(self.population, self.config.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            winners.append(winner)
        return winners
    
    def _crossover(self, parents: List[Prompt]) -> List[str]:
        """Perform LLM-based crossover between parent prompts."""
        offspring = []
        for _ in range(self.config.num_offspring):
            parent1, parent2 = random.sample(parents, 2)
            new_text = self.genetic_ops.crossover(parent1.text, parent2.text)
            
            # Ensure length constraints
            if len(new_text) > self.config.max_prompt_length:
                new_text = new_text[:self.config.max_prompt_length]
            offspring.append(new_text)
        
        return offspring
    
    def _mutate(self, text: str) -> str:
        """Apply LLM-based mutation operations to a prompt."""
        if random.random() > self.config.mutation_rate:
            return text
            
        operations = [
            self.genetic_ops.mutate,
            self.genetic_ops.insert_words,
            self.genetic_ops.swap_words,
            self.genetic_ops.substitute_words,
            self.genetic_ops.delete_words
        ]
        
        mutated_text = random.choice(operations)(text)
        
        # Ensure length constraints
        if len(mutated_text) > self.config.max_prompt_length:
            mutated_text = mutated_text[:self.config.max_prompt_length]
        elif len(mutated_text) < self.config.min_prompt_length:
            mutated_text += text[:(self.config.min_prompt_length - len(mutated_text))]
            
        return mutated_text
