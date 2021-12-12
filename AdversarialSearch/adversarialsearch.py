from typing import Callable

from AdversarialSearch.adversarialsearchproblem import (
    Action,
    AdversarialSearchProblem,
    State as GameState,
)


def maxValueMM(game : AdversarialSearchProblem[GameState, Action], state : GameState, player : int):
    # if game.isTerminal(state):
    if game.is_terminal_state(state):
        # return (game.utility(state, player), null)
        return (game.evaluate_terminal(state)[player], None)
    # (v, move) <- (-inf, null)
    (v, move) = (float('-inf'), None)
    # for each a in game.actions(state):
    for a in game.get_available_actions(state):
        # v2, a2 <- minValue(game, game.result(state, a))
        (v2, a2) = minValueMM(game, game.transition(state, a), player)
        # if (v2 > v):
        if v2 > v:
            # v, move <- v2, a
            (v, move) = (v2, a) # this might supposed to be a2
    # return (v, move)
    return (v, move)

def minValueMM(game: AdversarialSearchProblem[GameState, Action], state : GameState, player : int):
    # if game,isTerminal(state):
    if game.is_terminal_state(state):
        # return (game.utility(state, player), null)
        return (game.evaluate_terminal(state)[player], None)
    # (v, move) <- (inf, null)
    (v, move) = (float('inf'), None)
    # for each a in game.actions(state):
    for a in game.get_available_actions(state):
        # (v2, a2) <- maxValue(game, game.result(state, a))
        (v2, a2) = maxValueMM(game, game.transition(state, a), player)
        # if (v2 < v):
        if v2 < v:
            # (v, move) <- (v2, a)
            (v, move) = (v2, a)
    # return (v, move)
    return (v, move)

def minimax(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the minimax algorithm on ASPs, assuming that the given game is
    both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action (an element of asp.get_available_actions(asp.get_start_state()))
    """
    start = asp.get_start_state()
    # player <- game.toMove(state)
    player = start.player_to_move()
    # value, move <- maxValue(game, state)
    (value, move) = maxValueMM(asp, start, player)
    # return move
    return move



def maxValueAB(game: AdversarialSearchProblem[GameState, Action], state : GameState, alpha, beta, player):
    print("running max")
    # if game.isTerminal(state):
    if game.is_terminal_state(state):
        # return (game.Utility(state, player), null)
        return (game.evaluate_terminal(state)[player], None)
    # v <- -inf
    (v, move) = (float('-inf'), None)
    # for each a in game.actions(state):
    for a in game.get_available_actions(state):
        # (v2, a2) <- minValueAB(game, game.result(state, a), alpha, beta)
        (v2, a2) = minValueAB(game, game.transition(state, a), alpha, beta, player)
        # if (v2 > v):
        if v2 > v:
            # (v, move) <- (v2, a)
            (v, move) = (v2, a)
            # alpha <- max(alpha, v)
            alpha = max(alpha, v)
        # if (v >= beta):
        if v >= beta:
            # return (v, move)
            return (v, move)
    # return (v, move)
    return (v, move)

def minValueAB(game: AdversarialSearchProblem[GameState, Action], state : GameState, alpha, beta, player):
    print("running min")
    # if game.isTerminal(state):
    if game.is_terminal_state(state):
        # return (game.Utility(state, player), null)
        return (game.evaluate_terminal(state)[player], None)
    # v <- inf
    (v, move) = (float('inf'), None)
    # for each a in game.actions(state):
    for a in game.get_available_actions(state):
        # (v2, a2) <- maxValueAb(game, game.result(state, a), alpha, beta)
        (v2, a2) = maxValueAB(game, game.transition(state, a), alpha, beta, player)
        # if (v2 < v):
        if v2 < v:
            # (v, move) <- (v2, a)
            (v, move) = (v2, a)
            # beta <- min(beta, v)
            beta = min(beta, v)
        # if (v <= alpha):
        if v <= alpha:
            # return (v, move)
            return (v, move)
    # return (v, move)
    return (v, move)

def alpha_beta(asp: AdversarialSearchProblem[GameState, Action]) -> Action:
    """
    Implement the alpha-beta pruning algorithm on ASPs,
    assuming that the given game is both 2-player and constant-sum.

    Input:
        asp - an AdversarialSearchProblem
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    start = asp.get_start_state()
    # player <- game.toMove(state)
    player = start.player_to_move()
    # (value, move) <- maxValue(game, state, -inf, +inf)
    (value, move) = maxValueAB(asp, start, float('-inf'), float('inf'), player)
    # return move
    return move


def alpha_beta_cutoff(
    asp: AdversarialSearchProblem[GameState, Action],
    cutoff_ply: int,
    # See AdversarialSearchProblem:heuristic_func
    heuristic_func: Callable[[GameState], float],
) -> Action:
    """
    This function should:
    - search through the asp using alpha-beta pruning
    - cut off the search after cutoff_ply moves have been made.

    Input:
        asp - an AdversarialSearchProblem
        cutoff_ply - an Integer that determines when to cutoff the search and
            use heuristic_func. For example, when cutoff_ply = 1, use
            heuristic_func to evaluate states that result from your first move.
            When cutoff_ply = 2, use heuristic_func to evaluate states that
            result from your opponent's first move. When cutoff_ply = 3 use
            heuristic_func to evaluate the states that result from your second
            move. You may assume that cutoff_ply > 0.
        heuristic_func - a function that takes in a GameState and outputs a
            real number indicating how good that state is for the player who is
            using alpha_beta_cutoff to choose their action. You do not need to
            implement this function, as it should be provided by whomever is
            calling alpha_beta_cutoff, however you are welcome to write
            evaluation functions to test your implemention. The heuristic_func
            we provide does not handle terminal states, so evaluate terminal
            states the same way you evaluated them in the previous algorithms.
    Output:
        an action(an element of asp.get_available_actions(asp.get_start_state()))
    """
    start = asp.get_start_state()
    # player <- game.toMove(state)
    player = start.player_to_move()
    # (value, move) <- maxValue(game, state, -inf, +inf)
    depth = 1
    (value, move) = maxValueABCut(asp, start, float('-inf'), float('inf'), player, cutoff_ply, heuristic_func, depth)
    # return move
    return move

def maxValueABCut(game: AdversarialSearchProblem[GameState, Action], state : GameState, alpha, beta, player, cutoff, heur, depth):
    # if game.isTerminal(state):
    if game.is_terminal_state(state):
        # return (game.Utility(state, player), null)
        return (game.evaluate_terminal(state)[player], None)
    if depth == cutoff:
        return (game.heuristic_func(state, player), None)
    # v <- -inf
    (v, move) = (float('-inf'), None)
    # for each a in game.actions(state):
    for a in game.get_available_actions(state):
        new_depth = depth + 1
        # (v2, a2) <- minValueAB(game, game.result(state, a), alpha, beta)
        (v2, a2) = minValueABCut(game, game.transition(state, a), alpha, beta, player, cutoff, heur, new_depth)
        # if (v2 > v):
        if v2 > v:
            # (v, move) <- (v2, a)
            (v, move) = (v2, a)
            # alpha <- max(alpha, v)
            alpha = max(alpha, v)
        # if (v >= beta):
        if v >= beta:
            # return (v, move)
            return (v, move)
    # return (v, move)
    return (v, move)

def minValueABCut(game: AdversarialSearchProblem[GameState, Action], state : GameState, alpha, beta, player, cutoff, heur, depth):
    # if game.isTerminal(state):
    if game.is_terminal_state(state):
        # return (game.Utility(state, player), null)
        return (game.evaluate_terminal(state)[player], None)
    if depth == cutoff:
        return (game.heuristic_func(state, player), None)
    # v <- inf1
    (v, move) = (float('inf'), None)
    # for each a in game.actions(state):
    for a in game.get_available_actions(state):
        new_depth = depth + 1
        # (v2, a2) <- maxValueAb(game, game.result(state, a), alpha, beta)
        (v2, a2) = maxValueABCut(game, game.transition(state, a), alpha, beta, player, cutoff, heur, new_depth)
        # if (v2 < v):
        if v2 < v:
            # (v, move) <- (v2, a)
            (v, move) = (v2, a)
            # beta <- min(beta, v)
            beta = min(beta, v)
        # if (v <= alpha):
        if v <= alpha:
            # return (v, move)
            return (v, move)
    # return (v, move)
    return (v, move)
