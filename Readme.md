# Reinforcement Learning in Pacman Game

## Project Overview

This project explores the application of **Reinforcement Learning (RL)** to train an AI agent for the classic **Pacman** game. By implementing RL techniques, the project aims to enable Pacman to navigate through the maze, collect pellets, avoid ghosts, and maximize its score, demonstrating the power and adaptability of RL in game environments.

## Course Information

-   **Subject**: REL301m
-   **Class**: AI17B.DS
-   **Institution**: FPT University

## Project Goals

The primary objective of this project is to develop an AI agent that can make intelligent decisions in the Pacman game using reinforcement learning methods. Key goals include:

1.  Implementing various RL algorithms to train the Pacman agent.
2.  Testing and comparing the performance of each algorithm.
3.  Analyzing the learning process and outcomes to understand RL techniques better.

## Features

-   **Pacman Game Environment**: The project uses a simplified version of the Pacman game environment as the testing ground for RL models.
-   **Training with Reinforcement Learning**: Various RL techniques, including **SARSA with Function Approximation**, **Approximate Q Learning**, **Deep Q Learning**, are used to train Pacman to make optimal moves.
-   **Evaluation Metrics**: The agent’s performance is evaluated based on scores and win rate, demonstrating the efficiency of different RL approaches.

## Project Structure

-   **`/agents`**: Contains source code for the RL algorithms.
-   **`/utilities`**: Contains source code for the game.
-   **`pacman.py`**: Contains main code for running the game.
-   **`README.md`**: Project documentation.
-   **`requirements.txt`**: List of dependencies required to run the project.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the required libraries. Install dependencies using:

`pip install -r requirements.txt` 

### Running the Project

**Train the Agent and run the game**: Run `pacman.py` to start training the Pacman agent using the specified RL algorithm with specified game settings.
**Arguments**
-   **`-n` / `--numGames`**: _(int)_ Number of games to play. Default is 1.
    
-   **`-l` / `--layout`**: _(str)_ Layout file to load the map layout from. Default is `'mediumClassic'`.
    
-   **`-p` / `--pacman`**: _(str)_ Type of Pacman agent to use, specified in the `pacmanAgents` module. Default is `'KeyboardAgent'`.
    
-   **`-t` / `--textGraphics`**: Display output as text only (no graphics). Set as a flag (no value needed). Default is `False`.
    
-   **`-q` / `--quietTextGraphics`**: Generate minimal output with no graphics. Set as a flag. Default is `False`.
    
-   **`-g` / `--ghosts`**: _(str)_ Type of ghost agent to use, specified in the `ghostAgents` module. Default is `'RandomGhost'`.
    
-   **`-k` / `--numghosts`**: _(int)_ Maximum number of ghosts to use. Default is 4.
    
-   **`-z` / `--zoom`**: _(float)_ Zoom level for the graphics window. Default is `1.0`.
    
-   **`-f` / `--fixRandomSeed`**: Fix the random seed to make games deterministic (replayable). Set as a flag. Default is `False`.
    
-   **`-r` / `--recordActions`**: Record game histories to a file (named by the time they were played). Set as a flag. Default is `False`.
    
-   **`--replay`**: _(str)_ Replay a recorded game from a specified file (pickle format).
    
-   **`-a` / `--agentArgs`**: _(str)_ Comma-separated values for arguments passed to the agent (e.g., `"opt1=val1,opt2,opt3=val3"`).
    
-   **`-x` / `--numTraining`**: _(int)_ Number of episodes to use for training (suppresses output during training games). Default is `0`.
    
-   **`--frameTime`**: _(float)_ Delay time between frames in seconds; values `<0` mean keyboard input is expected. Default is `0.1`.
    
-   **`-c` / `--catchExceptions`**: Enable exception handling and timeouts during games. Set as a flag. Default is `False`.
    
-   **`--timeout`**: _(int)_ Maximum time (in seconds) an agent can spend computing in a single game. Default is `30`.
**Example Commands**

`python pacman.py -p ApproximateQLearningAgent -x 11 -n 10` 

## Algorithms Implemented

-   **SARSA with Function Approximation**: SARSA with function approximation learns an action-value function that generalizes to unseen states by approximating Q-values with a parameterized function, often beneficial in large or continuous state spaces.
-   **Approximate Q Learning**: This approach extends Q-learning by using a function approximator, such as a linear model or neural network, to estimate Q-values, allowing it to scale to more complex environments.
-   **Deep Q Learning**: Deep Q-Learning (DQN) uses a deep neural network to approximate the Q-values, enabling the agent to handle high-dimensional state spaces, such as image-based inputs in games.

## Results and Analysis

-   **Learning Curves**: Visualizes the agent's performance over time for each algorithm.
-   **Comparison of Algorithms**: Insights on the effectiveness and limitations of each approach.
-   **Challenges and Solutions**: Discussion on issues like exploration vs. exploitation and environment variability.

## Conclusion

This project demonstrates how reinforcement learning can be used to train an agent for a dynamic and complex game environment. With further tuning and advanced algorithms, the agent's performance could be enhanced significantly, making it a powerful application of RL in gaming.

## Future Works
-   **Advanced Algorithms for Diverse Layouts**: Future research could explore advanced RL algorithms like PPO, SAC, and DQN across various game layouts to better understand scalability and adaptability.
    
-   **Impact of Game Features**: Analyzing individual game features, such as obstacles or rewards, can reveal their effects on learning and help refine state representations for improved agent performance.
    
-   **Enhanced CNN Architectures**: Applying advanced CNN architectures, including attention and residual networks, could improve spatial learning and generalization in grid-based environments.

## Contribution
-  **Nguyễn Tấn Kiệt** - **QE170224**
-  **Đào Ngọc Huy** - **DE160024**
-  **Trương Trọng Tiến** - **QE170069**
-  **Phạm Quốc Hùng** - **QE170078**
-  **Phan Quốc Trung** - **QE170085**
-  **Diệp Gia Đông** - **QE170049**
