"""
The basic logic of the game of tic-tac-toe, including the management of the board,
the execution of moves, and the checking of victory conditions.
"""
class TicTacToe:
    """
    Represents a game of tic-tac-toe with basic board management and move execution.
    """
    def __init__(self):
        """
        Initialize the game with an empty board and player 1 starting.
        """
        self.board = [0] * 9  # 0 for empty, 1 for X, -1 for O
        self.current_player = 1  # Player 1 starts

    def move(self, idx):
        """
        Execute a move for the current player at the specified index.
        """
        if self.board[idx] == 0:
            self.board[idx] = self.current_player
            self.current_player = -self.current_player
            return True
        return False

    def check_winner(self):
        """
        Check if there is a winner on the current board configuration.
        """
        wins = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                (0, 3, 6), (1, 4, 7), (2, 5, 8),
                (0, 4, 8), (2, 4, 6)]
        for a, b, c in wins:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        if 0 not in self.board:
            return 0  # Draw
        return None  # Game continues

    def reset(self):
        """
        Reset the game to its initial state.
        """
        self.board = [0] * 9
        self.current_player = 1
