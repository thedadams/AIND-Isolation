"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
from random import randint
from collections import deque


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def close_to_center_score(game, player):
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    my_moves = game.get_legal_moves(player)
    if len(opponent_moves) == 0:
        return float("inf")
    if len(my_moves) == 0:
        return float("-inf")
    my_row, my_col = game.get_player_location(player)
    move_score = abs(game.height / 2 - my_row) + abs(game.width / 2 - my_col)
    return move_score


def far_from_center_score(game, player):
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    my_moves = game.get_legal_moves(player)
    if len(opponent_moves) == 0:
        return float("inf")
    if len(my_moves) == 0:
        return float("-inf")
    my_row, my_col = game.get_player_location(player)
    move_score = abs(game.height / 2 - my_row) + abs(game.width / 2 - my_col)
    return 1 / (move_score + 1)


def move_in_bounds(row, col, height, width):
    return row >= 0 and row < height and col >= 0 and col < width


def bfs_moves_scores(row, col, height, width, ratio):
    queue = deque()
    scores = [[0. for i in range(height)] for j in range(width)]
    queue.append((row, col, -1))
    while len(queue) != 0:
        r, c, s = queue.popleft()
        if not move_in_bounds(r, c, height, width) or scores[r][c] > 0.:
            continue
        s += 1
        scores[r][c] = ratio**(s)
        queue.append((r - 1, c + 2, s))
        queue.append((r - 1, c - 2, s))
        queue.append((r + 1, c - 2, s))
        queue.append((r + 1, c + 2, s))
        queue.append((r - 2, c - 1, s))
        queue.append((r - 2, c + 1, s))
        queue.append((r + 2, c - 1, s))
        queue.append((r + 2, c + 1, s))
    return scores


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    my_moves = game.get_legal_moves(player)
    if len(opponent_moves) == 0:
        return float("inf")
    if len(my_moves) == 0:
        return float("-inf")
    move_score = 0.
    my_row, my_col = game.get_player_location(player)
    opp_row, opp_col = game.get_player_location(game.get_opponent(player))
    my_row_reflect = my_col_reflect = opp_col_reflect = opp_row_reflect = False
    if my_row >= (game.height + 1) // 2:
        my_row = game.height - 1 - my_row
        my_row_reflect = True
    if my_col >= (game.width + 1) // 2:
        my_col = game.width - 1 - my_col
        my_col_reflect = True
    if opp_row >= (game.height + 1) // 2:
        opp_row = game.height - 1 - opp_row
        opp_row_reflect = True
    if opp_col >= (game.width + 1) // 2:
        opp_col = game.width - 1 - opp_col
        opp_col_reflect = True
    if (my_row, my_col) not in player.bfs_moves:
        player.bfs_moves[(my_row, my_col)] = bfs_moves_scores(my_row, my_col, game.height, game.width, 1. / 6.)
    if (opp_row, opp_col) not in player.bfs_moves:
        player.bfs_moves[(opp_row, opp_col)] = bfs_moves_scores(opp_row, opp_col, game.height, game.width, 1. / 6.)
    my_scores = player.bfs_moves[(my_row, my_col)]
    opp_scores = player.bfs_moves[(opp_row, opp_col)]
    for i, j in game.get_blank_spaces():
        r = game.height - 1 - i if my_row_reflect else i
        c = game.width - 1 - j if my_col_reflect else j
        move_score += my_scores[r][c]
        r = game.height - 1 - i if opp_row_reflect else i
        c = game.width - 1 - j if opp_col_reflect else j
        move_score -= opp_scores[r][c]
    return move_score


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate successors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=15.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.bfs_moves = dict()

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if len(legal_moves) > 8:
            if game.width * game.height == len(legal_moves):
                return (randint(0, game.height - 1), randint(0, game.width - 1))
            else:
                return self.second_move(game)
        if len(legal_moves) == 0:
            return (-1, -1)
        next_move = (-1, -1)
        if self.iterative:
            self.search_depth = 1
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            while True:
                if self.method == "alphabeta":
                    _, next_move = self.alphabeta(game, self.search_depth)
                else:
                    _, next_move = self.minimax(game, self.search_depth)
                if not self.iterative:
                    break
                self.search_depth += 1
        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass
        # Return the best move from the last completed search iteration
        return next_move

    def second_move(self, game):
        row, col = game.get_player_location(game.get_opponent(self))
        if row < game.width // 2:
            row += 1
        else:
            row -= 1
        return (row, col)

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if maximizing_player:
            return self.max_value(game, depth)
        else:
            return self.min_value(game, depth)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if maximizing_player:
            return self.max_value(game, depth, True, alpha, beta)
        else:
            return self.min_value(game, depth, True, alpha, beta)

    def max_value(self, game, depth, alphabeta=False, alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), (-1, -1)
        utility = float("-inf")
        next_move = (-1, -1)
        for move in game.get_legal_moves(self):
            this_score, _ = self.min_value(game.forecast_move(
                move), depth - 1, alphabeta, alpha, beta)
            if utility < this_score:
                utility = this_score
                next_move = move
            if alphabeta:
                if utility >= beta:
                    break
                alpha = max(alpha, utility)
        return utility, next_move

    def min_value(self, game, depth, alphabeta=False, alpha=float("-inf"), beta=float("inf")):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), (-1, -1)
        utility = float("inf")
        next_move = (-1, -1)
        for move in game.get_legal_moves(game.get_opponent(self)):
            this_score, _ = self.max_value(game.forecast_move(
                move), depth - 1, alphabeta, alpha, beta)
            if utility > this_score:
                utility = this_score
                next_move = move
            if alphabeta:
                if utility <= alpha:
                    break
                beta = min(beta, utility)
        return utility, next_move
