"""
The saved models are loaded using the tensorflow.keras.models.load_model method.
"""


import os
import numpy as np
from keras.models import load_model


# load_model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "train", "tic_tac_toe_model.h5")
model = load_model(model_path)
print("Model loaded successfully from:", model_path)

# Define a simple function to use the model
def predict_move(previous_board, current_board):
    """
    Predict the best move position for the next step based on
    the previous chessboard state and the current chessboard state.
    """
    previous_board = np.array(previous_board).reshape(1, 9)
    probabilities = model.predict(previous_board)[0]
    valid_moves = [i for i, v in enumerate(current_board) if v == 0]
    probabilities = np.array([probabilities[i] if i in valid_moves else 0 for i in range(9)])
    recommended_action = np.argmax(probabilities)
    return recommended_action

# Example board
example_board = [0, 0, 0, 1, -1, 0, 0, 1, -1]  # 0 means empty, 1 means X, -1 means O

# Predict next step
action = predict_move(example_board, example_board)
print(f"Recommended move for the board {example_board} is at position: {action}")
