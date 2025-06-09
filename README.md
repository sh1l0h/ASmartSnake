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

## Controls
- Use arrow keys to control the snake manually
- Press 'Q' to quit the game
- Press 'R' to restart the game
- Press 'P' to pause/unpause
