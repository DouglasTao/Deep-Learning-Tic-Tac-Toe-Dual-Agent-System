"""
Responsible for training and evaluating the intelligences,
as well as controlling the operation of the game.
"""


from tic_tac_toe_env import TicTacToe
from tic_tac_toe_agent import Agent
import os
import time

def train_agent(episodes):
    """
    Train the agent for a specified number of episodes.
    """
    game = TicTacToe()
    agent = Agent()

    start_time = time.time()  # Record start time

    for episode in range(episodes):
        game.reset()
        done = False
        while not done:
            state = game.board.copy()
            action = agent.choose_action(state)
            game.move(action)
            reward = game.check_winner()
            agent.train([state], [action], [reward])
            done = reward is not None

        if episode % 10 == 0:  # Print every 10 episodes to reduce clutter
            print(f"Episode {episode + 1}/{episodes} completed.")

    end_time = time.time()  # Record end time
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    # Save the model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_directory = os.path.join(current_dir, "train")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_path = os.path.join(model_directory, "tic_tac_toe_model.h5")
    agent.save_model(model_path)

    print("Training completed.")

if __name__ == "__main__":
    train_agent(1)  # Adjust as necessary
