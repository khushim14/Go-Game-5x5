# Go Game AI – 5x5 Board

🎯 **Goal:** Implement a playable 5x5 Go game AI using Minimax with Alpha-Beta Pruning and Heuristics.

## 📌 Features

- Handles basic Go rules (liberties, suicide, KO)
- 5x5 board with custom scoring heuristics
- Uses iterative deepening + alpha-beta pruning
- Evaluates board state with territory, liberties, and position
- Supports `PASS` move

## 🧠 AI Strategy

- Minimax search with a depth of 4
- Transposition table for efficiency
- Evaluation includes:
  - Number of stones
  - Liberty count
  - Territory estimation
  - Group penalty
  - Positional bonus (center/edge/corner)

## 🛠 Tech Stack

`Python 3` · `NumPy` · `Hashlib`

## 🧪 How to Run

1. Place your `input.txt` file in the directory. Format:

2. Run the game:
```bash
python code.py

3. Move is written to output.txt in format: i,j or PASS

## 📂 Files

- `code.py`: Main agent that plays Go using Minimax + Alpha-Beta pruning
- `input.txt`: Input file with player color, previous and current board state (required)
- `output.txt`: Output file containing the move made by the agent



