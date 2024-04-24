"""
Responsible for training and evaluating the intelligences,
as well as controlling the operation of the game.
"""


import os
import time
import numpy as np
from tic_tac_toe_env import TicTacToe
from tic_tac_toe_agent import Agent

def initialize_game():
    """
    Initialize the Tic Tac Toe game and agent.
    """
    game = TicTacToe()
    agent = Agent()
    return game, agent

def play_episode(game, agent, win_count, loss_count, draw_count):
    """
    Play a single episode of the game using the provided game environment and agent.
    """
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
            done = True
            if reward == 1:
                win_count += 1
            elif reward == -1:
                loss_count += 1
            elif reward == 0:
                draw_count += 1
    loss_history = agent.train(states, actions, rewards)
    return win_count, loss_count, draw_count, loss_history

def save_model(agent, directory="train", filename="tic_tac_toe_model.h5"):
    """
    Saves the trained model to a file in the specified directory.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_directory = os.path.join(current_dir, directory)
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_path = os.path.join(model_directory, filename)
    agent.save_model(model_path)

def train_agent(episodes):
    """
    Train the agent over a specified number of episodes
    and log loss values to a file using UTF-8 encoding.
    """
    game, agent = initialize_game()
    win_count, loss_count, draw_count = 0, 0, 0
    start_time = time.time()

    # Open a file to write loss histories with UTF-8 encoding
    with open("train/loss_history.txt", "w", encoding='utf-8') as file:
        file.write("Episode,Average Loss\n")

        for episode in range(episodes):
            win_count, loss_count, draw_count, loss_history = \
            play_episode(game, agent, win_count, loss_count, draw_count)
            average_loss = np.mean(loss_history) if loss_history else 0
            file.write(f"{episode + 1},{average_loss:.4f}\n")
            print(f"Episode {episode + 1}/{episodes} completed "
               f"with average loss {average_loss:.4f}")

        total_games = win_count + loss_count + draw_count
        win_rate = win_count / total_games if total_games != 0 else 0
        print(f"Win rate: {win_rate:.2f}, Training time: {time.time() - start_time:.2f} seconds")

        save_model(agent)

        print("Training completed.")


if __name__ == "__main__":
    while True:
        try:
            num_episodes = int(input("Enter the number of episodes to train: "))
            if num_episodes <= 0:
                raise ValueError
            break
        except ValueError:
            print("Please enter a valid positive integer.")

    train_agent(num_episodes)
