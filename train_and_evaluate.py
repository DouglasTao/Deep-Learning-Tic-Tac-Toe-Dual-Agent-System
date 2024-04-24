"""
Responsible for training and evaluating the intelligences,
as well as controlling the operation of the game.
"""


import os
import time
import numpy as np
from tic_tac_toe_env import TicTacToe
from tic_tac_toe_agent import Agent


def train_agent(episodes):
    """
    Train the agent for a specified number of episodes.
    """
    game = TicTacToe()
    agent = Agent()
    win_count, loss_count, draw_count = 0, 0, 0

    start_time = time.time()

    for episode in range(episodes):
        game.reset()
        done = False
        states, actions, rewards = [], [], []
        while not done:
            state = game.board.copy()
            action = agent.choose_action(state)
            valid_move = game.move(action)
            print(f"Action chosen: {action}, Valid move: {valid_move}, Board: {game.board}")
            if not valid_move:
                continue
            reward = game.check_winner()
            print(f"Reward: {reward}, Game Done: {done}")
            if reward is not None:
                states.append(state)
                actions.append(action)
                rewards.append(reward if reward == game.current_player else -reward)
                done = True  # This sets done when a reward is determined
                if reward == 1:
                    win_count += 1
                elif reward == -1:
                    loss_count += 1
                elif reward == 0:
                    draw_count += 1

        loss_history = agent.train(states, actions, rewards)
        if episode % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} completed with average loss {np.mean(loss_history):.4f}")

    total_games = win_count + loss_count + draw_count
    win_rate = win_count / total_games if total_games != 0 else 0
    print(f"Win rate: {win_rate:.2f}, Training time: {time.time() - start_time:.2f} seconds")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_directory = os.path.join(current_dir, "train")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_path = os.path.join(model_directory, "tic_tac_toe_model.h5")
    agent.save_model(model_path)

    print("Training completed.")


if __name__ == "__main__":
    train_agent(1)  # Adjust as necessary
