"""
The tic-tac-toe intelligences, including the neural network model,
the logic for selecting moves, and the training methodology.
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input


class Agent:
    def __init__(self):
        self.model = Sequential([
            Input(shape=(9,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(9, activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')

    def predict(self, board):
        board = np.array(board).reshape(1, 9)
        probabilities = self.model.predict(board)[0]
        return probabilities

    def choose_action(self, state):
        probabilities = self.predict(state)
        probabilities[state != 0] = 0  # Ensure no action on filled cells
        action = np.argmax(probabilities)
        return action

    def train(self, states, actions, rewards):
        for state, action, reward in zip(states, actions, rewards):
            target = self.predict(state)
            target[action]=state[action]
            self.model.fit(np.array([state]), np.array([target]), epochs=1, verbose=0)

    def save_model(self, file_path):
        self.model.save(file_path)
        print(f"Model saved to {file_path}.")
