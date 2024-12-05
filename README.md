# Prompt Evolution

A Python package for optimizing prompts using evolutionary algorithms. This implementation allows for flexible configuration of the evolution process and can be easily adapted to different prompt optimization scenarios.

## Features

- Configurable evolutionary algorithm parameters
- Multiple genetic operators (mutation, crossover)
- Tournament selection for parent selection
- Elitism to preserve best solutions
- Early stopping to prevent overfitting
- Customizable fitness function
- Progress tracking and history logging

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

2. Import and use the package:
```python
from prompt_evolution import PromptEvolution, EvolutionConfig

# Configure evolution parameters
config = EvolutionConfig(
    population_size=20,
    num_generations=10,
    mutation_rate=0.3,
    crossover_rate=0.7
)

# Create evolution instance
evolution = PromptEvolution(config)

# Initialize population with base prompt
evolution.initialize_population("Your initial prompt here")

# Define fitness function
def evaluate_prompt(prompt: str) -> float:
    # Implement your evaluation logic
    return score

# Run evolution
best_prompt = evolution.evolve(evaluate_prompt)
```

See `example.py` for a complete usage example.

## Configuration Options

The `EvolutionConfig` class supports the following parameters:

- `population_size`: Size of the prompt population
- `num_generations`: Number of generations to evolve
- `mutation_rate`: Probability of mutation (0-1)
- `crossover_rate`: Probability of crossover (0-1)
- `elite_size`: Number of best prompts to preserve
- `tournament_size`: Size of tournament for selection
- `max_prompt_length`: Maximum length of prompts
- `min_prompt_length`: Minimum length of prompts
- `patience`: Generations without improvement before early stopping
- `save_history`: Whether to save evolution history
- `verbose`: Whether to print progress information

## Customization

You can customize the evolution process by:

1. Implementing your own fitness function
2. Modifying genetic operators in the `PromptEvolution` class
3. Adjusting configuration parameters
4. Adding new mutation operations

## License

MIT License
