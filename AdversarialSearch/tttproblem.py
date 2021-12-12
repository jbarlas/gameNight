# author Vincent Kubala; GameUI implementation by Eli Zucker

###############################################################################
# State and Action Representations:
#
# - A (game) state is a TTTState that contains game board and the index of the
#   player to move
#
# - An action is a pair that represents the location at which a player
#   would like to draw an X or an O: (<index of row>, <index of column>),
#   where the indices start at 0.
#
# - See the main function at the bottom of this file for an example.
#
###############################################################################


from typing import Tuple
from adversarialsearchproblem import AdversarialSearchProblem, GameState, GameUI
import time
import math

SPACE = " "
X = "X"  # Player 0 is X
O = "O"  # Player 1 is O
PLAYER_SYMBOLS = [X, O]


class TTTState:
    def __init__(self, board, ptm):
        """
        Inputs:
                board - represented as a 2D List of character strings.
                Each character in the board is X, O, or SPACE (see above
                for global definition), where SPACE indicates that the
                corresponding cell of the tic-tac-toe board is empty.

                ptm- the index of the player to move, which will be 0 or 1,
                where 0 corresponds to the X player, who moves first, and
                1 to the O player, who moves second.
        """
        self.board = board
        self.ptm = ptm

    def player_to_move(self):
        return self.ptm


# In TTT, an action consists of placing a piece on a 2D grid. Thus, our actions need two pieces of
# data: both row and column. So the type of our action is tuple with two ints.
Action = Tuple[int, int]

def sigmoid(x):
    val = 1 / (1 + math.e ** (-x))
    return val

class TTTProblem(AdversarialSearchProblem[TTTState, Action]):
    def __init__(self, dim=3, board=None, player_to_move=0):
        """
        Inputs:
                dim- the number of cells in one row or column.
                board - 2d list of character strings (as in TTTState)
                player_to_move- index of player to move (as in TTTState).

                The board and player_to_move together constitute the start state
                of the game
        """
        self._dim = dim
        if board == None:
            board = [[SPACE for _ in range(dim)] for _ in range(dim)]
        self._start_state = TTTState(board, player_to_move)

    def heuristic_func(self, state: TTTState, player_index: int) -> float:
        """
        TODO: Fill this out with your own heuristic function! You should make sure that this
        function works with boards of any size; if it only works for 3x3 boards, you won't be
        able to properly test ab-cutoff for larger board sizes!
        """
        player_index += 1
        opp_index = player_index % 2 + 1
        board = state.board
        dim = len(board)
        score = 0

        # Score all possible connect 4 "slices"
        for slice in self.get_slices(state):
            if slice.count(player_index) == dim - 1 and slice.count(0) == 1:
                score += 20
            elif slice.count(player_index) == dim - 2 and slice.count(0) == 2:
                score += 10
            elif slice.count(opp_index) == dim - 1 and slice.count(0) == 1:
                score -= 20
            elif slice.count(player_index) > 0 and slice.count(opp_index) > 0:
                score += 0
        score = sigmoid(score)
        return score

    def get_slices(self, state):
        board = state.board
        rows = len(board)
        slices = []
        diagonal1 = []
        diagonal2 = []
        for index in range(0, rows):
            slices.append(board[index][:])
            slices.append(board[:][index])
            diagonal1.append(board[index][index])
            diagonal2.append(board[rows-index-1][index])
        slices.append(diagonal1)
        slices.append(diagonal2)
        return slices

    def get_available_actions(self, state):
        actions = set()
        for r in range(self._dim):
            for c in range(self._dim):
                if state.board[r][c] == " ":
                    actions.add((r, c))
        return actions

    def transition(self, state, action):
        assert not (self.is_terminal_state(state))
        assert action in self.get_available_actions(state)

        # make deep copy of board
        board = [[elt for elt in row] for row in state.board]

        board[action[0]][action[1]] = PLAYER_SYMBOLS[state.ptm]
        return TTTState(board, 1 - state.ptm)

    def is_terminal_state(self, state):
        return not (self._internal_evaluate_terminal(state) == "non-terminal")

    def evaluate_terminal(self, state):
        internal_val = self._internal_evaluate_terminal(state)
        if internal_val == "non-terminal":
            raise ValueError("attempting to evaluate a non-terminal state")
        else:
            return internal_val

    def _internal_evaluate_terminal(self, state):
        """
        If state is terminal, returns its evaluation;
        otherwise, returns 'non-terminal'.
        """
        board = state.board

        diagonal1 = [board[i][i] for i in range(self._dim)]
        # bind the output of _all_same for diagonal1 for its two uses
        asd1 = TTTProblem._all_same(diagonal1)
        if asd1:  # #onlyinpython / #imissoptions
            return asd1

        diagonal2 = [board[i][self._dim - 1 - i] for i in range(self._dim)]
        asd2 = TTTProblem._all_same(diagonal2)
        if asd2:
            return asd2

        for row in board:
            asr = TTTProblem._all_same(row)
            if asr:
                return asr

        for c in range(self._dim):
            # why oh why didn't I just use numpy arrays?
            col = [board[r][c] for r in range(self._dim)]
            asc = TTTProblem._all_same(col)
            if asc:
                return asc

        if self.get_available_actions(state) == set():
            # all spaces are filled up
            return [0.5, 0.5]
        else:
            return "non-terminal"

    @staticmethod
    def _all_same(cell_list):
        """
        Given a list of cell contents, e.g. ['x', ' ', 'X'],
        returns [1.0, 0.0] if they're all X, [0.0, 1.0] if they're all O,
        and False otherwise.
        """
        xlist = [cell == X for cell in cell_list]
        if all(xlist):
            return [1.0, 0.0]

        olist = [cell == O for cell in cell_list]
        if all(olist):
            return [0.0, 1.0]

        return False

    @staticmethod
    def board_to_pretty_string(board):
        """
        Takes in a tile game board and outputs a pretty string representation
        of it for printing.
        """
        hbar = "-"
        vbar = "|"
        corner = "+"
        dim = len(board)

        s = corner
        for _ in range(2 * dim - 1):
            s += hbar
        s += corner + "\n"

        for r in range(dim):
            s += vbar
            for c in range(dim):
                s += str(board[r][c]) + " "
            s = s[:-1]
            s += vbar
            s += "\n"

        s += corner
        for _ in range(2 * dim - 1):
            s += hbar
        s += corner
        return s


# Basic TTT GameUI implementation (prints board states to console)
class TTTUI(GameUI):
    def __init__(self, asp: TTTProblem, delay=0.2):
        self._asp = asp
        self._delay = delay
        self._state = TTTState(
            [[SPACE for _ in range(asp._dim)] for _ in range(asp._dim)], 0
        )  # empty state

    def render(self):
        print(TTTProblem.board_to_pretty_string(self._state.board))
        time.sleep(self._delay)

    def get_user_input_action(self):
        """
        Output- Returns an action obtained through the GameUI input itself.
        """
        user_action = None
        available_actions = self._asp.get_available_actions(self._state)

        while not user_action in available_actions:
            row = int(input("Enter row index: "))
            col = int(input("Enter column index: "))
            user_action = (row, col)

        return user_action


def main():
    """
    Provides an example of the TTTProblem class being used.
    """
    t = TTTProblem()
    # A state in which an X is in the center cell, and O moves next.
    s0 = TTTState([[" ", " ", " "], [" ", "X", " "], [" ", " ", " "]], 1)
    # The O player puts down an O at the top-left corner, and now X moves next
    s1 = t.transition(s0, (0, 0))
    assert s1.board == [["O", " ", " "], [" ", "X", " "], [" ", " ", " "]]
    assert s1.ptm == 0


if __name__ == "__main__":
    main()
