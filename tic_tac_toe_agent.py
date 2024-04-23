"""
The tic-tac-toe intelligences, including the neural network model,
the logic for selecting moves, and the training methodology.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input


class Agent:
    """
    Represents an agent that plays tic-tac-toe using a neural network model.
    """
    def __init__(self):
        """
        Initializes the Agent with a neural network model.
        """
        self.model = Sequential([
            Input(shape=(9,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(9, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def predict(self, board):
        """
        Predicts the probabilities of each action for the given board state.
        """
        board = np.array(board).reshape(1, 9)
        probabilities = self.model.predict(board)[0]
        return probabilities

    def choose_action(self, state):
        """
        Chooses the best action to take based on the current board state.
        """
        probabilities = self.predict(state)
        probabilities[state != 0] = 0  # Ensure no action on filled cells
        action = np.argmax(probabilities)
        return action

    def train(self, states, actions, rewards):
        """
        Trains the neural network model using the given states, actions, and rewards.
        """
        history_loss = []
        for state, action, reward in zip(states, actions, rewards):
            target = self.predict(state)
            target[action]=state[action]
            history = self.model.fit(np.array([state]), np.array([target]), epochs=1, verbose=0)
            history_loss.append(history.history['loss'][0])
        return history_loss

    def save_model(self, file_path):
        """
        Saves the neural network model to a file.
        """
        self.model.save(file_path)
        print(f"Model saved to {file_path}.")

    def get_latest_loss(self):
        """
        Retrieves the latest loss value from the model training process.
        """
        _, latest_loss = self.model.fit(np.zeros((1, 9)), np.zeros((1, 9)), verbose=0)
        return latest_loss
