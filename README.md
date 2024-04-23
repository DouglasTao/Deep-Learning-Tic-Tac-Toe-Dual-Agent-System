# Deep Learning-based Tic-Tac-Toe Dual-Agent System

This project develops a dual-agent system for playing Tic-Tac-Toe using deep learning techniques. The system models the game state and learns adversarial strategies through interactions between two intelligent agents. This README outlines the system's architecture, inputs, training process, and strategic adaptation mechanisms.

## System Overview

The Tic-Tac-Toe system incorporates a comprehensive approach to model the game environment and employs a deep learning architecture to enable two agents to learn and adapt their strategies through gameplay. The agents compete against each other, enhancing their abilities and adjusting their tactics based on the game's outcome and their own performance metrics.

### System Inputs and Structure

- **Board State Inputs**: The system uses a series of 9-element vectors to represent the state of the Tic-Tac-Toe board:
  - Black pieces' positions
  - White pieces' positions
  - Black's current positions
  - White's current positions
- **Action Units**: There are five action units representing potential moves: up, down, left, right, and place. These units can either represent potential actions to be taken by the agents or the last action performed.

### State and Feedback

- **State Unit**: The system maintains a state unit that represents the current win probability, which is computed based on the prevailing board configuration. A function triggered by a low win probability state recalculates model parameters to improve strategy, increasing the error margin deliberately to force learning and adaptation.

### Model Architecture

- **Configuration**: The system is vaguely described as having a "42*m*n*42" architecture, suggesting a multi-layered approach where `42` might represent a specific feature dimension, and `m` and `n` signify the number of layers or types of layers used.
- **Sequential Data Processing**: LSTM (Long Short-Term Memory) or RNN (Recurrent Neural Network) layers are utilized to manage the game's sequential nature, reflecting each move's dependence on the previous state of the board.

### Training Process and Error Handling

- **Initial Training Phase**: Focuses on reducing misunderstandings about the game state, i.e., enhancing the prediction accuracy of the board status.
- **Error Shifting**: As training progresses, errors are increasingly propagated into the state function, which dictates how strategies are adjusted based on the current game status.
- **Adaptive Learning**: The system displays an adaptive learning behavior where an agent, upon frequently losing, shifts its strategy to increase its win probability, thus maintaining a competitive edge.

## Feasibility and Implementation

Theoretically, this system is viable, particularly for a relatively simple game like Tic-Tac-Toe. The critical challenge in practical implementation lies in designing effective state evaluation functions and intelligent strategy adjustment mechanisms that enable the two agents to competitively enhance their capabilities over time.

## Running the System

To run this system, ensure that all prerequisites are installed and then execute the main training script:

```bash
python train_and_evaluate.py
```

This script will initiate the training process, where the two agents play against each other, learning and adapting their strategies based on the outcomes and the feedback mechanism designed.

## Dependencies

- TensorFlow
- NumPy

Make sure to install these dependencies using pip:

```bash
pip install tensorflow numpy
```

## Conclusion

This deep learning-based dual-agent system for Tic-Tac-Toe represents a sophisticated approach to machine learning in games, focusing on continuous learning and strategy adaptation. The system's effectiveness will depend significantly on the refinement of its architecture and the efficiency of its training processes.
