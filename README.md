# A Smart Snake

## Team
Giorgi Sirdadze
Luka Gorgadze
Luka Trapaidze
Mirian Shilakadze

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Running the Simulation

1. To run the game with the Neural Network agent:
```bash
python src/display_game.py --agent nn
```

2. To run the game with the Q-Learning agent:
```bash
python src/display_game.py --agent q
```

3. To run the game with the A* Search agent:
```bash
python src/display_game.py --agent astar
```

## Training the Agents

1. To train the Neural Network agent:
```bash
python src/nn_agent.py
```

2. To train the Q-Learning agent:
```bash
python src/q_agent.py
```

## Comparing Agents

To run a comprehensive comparison between all three agents:
```bash
python src/compare_agents.py
```

## Storage and Output Files

After running the various commands, training data and visualizations are stored in the `storage/` directory:

### Neural Network Training (`python src/nn_agent.py`)
- **Trained weights**: `storage/nn/snake_weights_{width}x{height}.pth`
- **Training progress plots**: `storage/nn/training_progress.png` (updated every 50 episodes)

### Q-Learning Training (`python src/q_agent.py`)
- **Q-table data**: `storage/q/snake_table_{width}x{height}.pkl`
- **Training progress plots**: `storage/q/training_progress.png` (updated every 50 episodes)

### Agent Comparison (`python src/compare_agents.py`)
- **Comparison dashboard**: `storage/compare/snake_ai_comparison.png`

## Controls
- Use arrow keys to control the snake manually
- Press 'Q' to quit the game
- Press 'R' to restart the game
- Press 'P' to pause/unpause

## Directory Structure
```
storage/
├── nn/                                    # Neural Network artifacts
│   ├── snake_weights_{width}x{height}.pth # Trained model weights
│   └── training_progress.png             # Training progress visualization
├── q/                                     # Q-Learning artifacts
│   ├── snake_table_{width}x{height}.pkl  # Trained Q-table
│   └── training_progress.png             # Training progress visualization
└── compare/                               # Comparison results
    └── snake_ai_comparison.png            # Agent performance comparison
```
