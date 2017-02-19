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


def far_from_center_score(game, player):
    """
    This heuristic values squares furthest from the center of the board.

    Parameters
    ----------
        game - a Board object representing the current state of the game.
        player - a CustomPlayer object that represents the player using this heuristic

    Returns
    -------
        float - heuristic value of the current move
    """
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    my_moves = game.get_legal_moves(player)
    if len(opponent_moves) == 0:
        return float("inf")
    if len(my_moves) == 0:
        return float("-inf")
    my_row, my_col = game.get_player_location(player)
    opp_row, opp_col = game.get_player_location(game.get_opponent(player))
    score = abs(game.height / 2 - my_row) + abs(game.width / 2 - my_col)
    score -= abs(game.height / 2 - opp_row) + abs(game.width / 2 - opp_col)
    return score


def close_to_center_score(game, player):
    """
    This heuristic values squares in the center of the board more than those at the edge of the board.

    Parameters
    ----------
        game - a Board object representing the current state of the game.
        player - a CustomPlayer object that represents the player using this heuristic

    Returns
        float - heuristic value of the current move
    """
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    my_moves = game.get_legal_moves(player)
    if len(opponent_moves) == 0:
        return float("inf")
    if len(my_moves) == 0:
        return float("-inf")
    my_row, my_col = game.get_player_location(player)
    opp_row, opp_col = game.get_player_location(game.get_opponent(player))
    score = -(abs(game.height / 2 - my_row) + abs(game.width / 2 - my_col))
    score -= -(abs(game.height / 2 - opp_row) + abs(game.width / 2 - opp_col))
    return score


def move_in_bounds(row, col, height, width):
    """
    This is a helper function to make sure we are in the board bounds.
    """
    return row >= 0 and row < height and col >= 0 and col < width


def bfs_moves_scores(row, col, height, width, blank_spaces):
    """
    This helper function calculates the shortest path to each square on the board.
    It then scores each square as 1/(s+1), where s is the number of steps of the shortest path to that square from (row, col)
    using only spaces that are blank.

    Parameters
    ----------
    (row, col) - the position to start from
    height - number of rows in the board
    width - number of columns in the board
    blank_spaces - a set of tuples containing all the blank spaces on the board

    Returns
    -------
    float - the score for the player with the given game state
    """
    queue = deque()
    visited = set()
    score = 0.
    queue.append(((row, col), 1))
    while len(queue) != 0:
        (r, c), s = queue.popleft()
        next_moves = []
        for i in [-1, 1]:
            for j in [-2, 2]:
                # For each possible move, we check that we are in bounds,
                # the space is blank, and we haven't already visited it.
                if move_in_bounds(r + i, c + j, height, width) and (r + i, c + j) in blank_spaces and (r + i, c + j) not in visited:
                    next_moves.append((r + i, c + j))
                    visited.add((r + i, c + j))
                if move_in_bounds(r + j, c + i, height, width) and (r + j, c + i) in blank_spaces and (r + j, c + i) not in visited:
                    next_moves.append((r + j, c + i))
                    visited.add((r + j, c + i))
        score += 1. / s
        queue.extend(zip(next_moves, [s + 1 for i in range(len(next_moves))]))
    return score


def bfs_heuristic(game, player):
    """
    This heuristic user BFS to calculate a score for each square on the board for each player in the game.
    Then it adds up all the scores of the blank squares for `player` and subtracts the score of each blank
    square for `opponent`.
    This idea is that it is a generalization of `improved_score`: `improved_score` does this for the next moves,
    but here we use all possible remaining moves.

    Parameters
    ----------
    game - a Board object representing the current state of the game.
    player - a CustomPlayer object that represents the player using this heuristic

    Returns
    -------
    float - the heuristic value of the current move
    """
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    my_moves = game.get_legal_moves(player)
    # If we know whether we have won or lost, determine that now.
    if len(opponent_moves) == 0:
        return float("inf")
    if len(my_moves) == 0:
        return float("-inf")
    move_score = 0.
    my_row, my_col = game.get_player_location(player)
    opp_row, opp_col = game.get_player_location(game.get_opponent(player))
    blank_spaces = set(game.get_blank_spaces())
    move_score = bfs_moves_scores(my_row, my_col, game.height, game.width, blank_spaces)
    move_score -= bfs_moves_scores(opp_row, opp_col, game.height, game.width, blank_spaces)
    return move_score


def far_from_opponent_score(game, player):
    """
    This heuristic chooses the move that is furtherest from the opponent.

    Parameters
    ----------
    game - a Board object representing the current state of the game.
    player - a CustomPlayer object that represents the player using this heuristic

    Returns
    -------
    float - the heuristic value of the current move
    """
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    my_moves = game.get_legal_moves(player)
    if len(opponent_moves) == 0:
        return float("inf")
    if len(my_moves) == 0:
        return float("-inf")
    my_row, my_col = game.get_player_location(player)
    opp_row, opp_col = game.get_player_location(game.get_opponent(player))
    return float(abs(my_row - opp_row) + abs(my_col - opp_col))


def close_to_opponent_score(game, player):
    """
    This heuristic chooses the move that is closest to the opponent.

    Parameters
    ----------
    game - a Board object representing the current state of the game.
    player - a CustomPlayer object that represents the player using this heuristic

    Returns
    -------
    float - the heuristic value of the current move
    """
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    my_moves = game.get_legal_moves(player)
    if len(opponent_moves) == 0:
        return float("inf")
    if len(my_moves) == 0:
        return float("-inf")
    my_row, my_col = game.get_player_location(player)
    opp_row, opp_col = game.get_player_location(game.get_opponent(player))
    return -float(abs(my_row - opp_row) + abs(my_col - opp_col))


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
    float - The heuristic value of the current game state to the specified player.
    """
    return close_to_opponent_score(game, player) + bfs_heuristic(game, player)


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
    ratio : float (optional)
        The ratio used to calculate the BFS heuristic for the player.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, iterative=True, method='minimax', timeout=10., ratio=0.5):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        # This is used to store the board move scores in the BFS heuristic.
        self.bfs_moves = dict()
        self.ratio = ratio

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

        # If we are picking our position on the board.n
        if len(legal_moves) > 8:
            # If we are the first to pick.
            if game.width * game.height == len(legal_moves):
                # We pick a random spot.
                return (randint(0, game.height - 1), randint(0, game.width - 1))
            else:
                # If we are second to pick, then we pick one column inside the first player, if we can.
                row, col = game.get_player_location(game.get_opponent(self))
                if col < game.width // 2:
                    col += 1
                else:
                    col -= 1
                    return (row, col)

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
                # If we are not using iterative deepening, then we exit this loop after the first run.
                if not self.iterative:
                    break
                self.search_depth += 1
        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass
        # Return the best move from the last completed search iteration
        return next_move

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
        """
        This handles the max level of the game tree.

        Parameters
        ----------
        game - a Board object representing the current game state
        depth - the depth we have left to search
        alphabeta - bool, True if we are using alphabeta pruning
        alpha - the lower bound for alphabeta
        beta - the upper bound for alphabeta

        Returns
        -------
        float - the score for the current search branch
        tuple(int, int) - the best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), (-1, -1)
        utility = float("-inf")
        next_move = (-1, -1)
        for move in game.get_legal_moves(self):
            this_score, _ = self.min_value(game.forecast_move(move), depth - 1, alphabeta, alpha, beta)
            # If we found a better score, we update it.
            if utility < this_score:
                utility = this_score
                next_move = move
            if alphabeta:
                # We prune, as in we do not need to calculate further.
                if utility >= beta:
                    break
                alpha = max(alpha, utility)
        return utility, next_move

    def min_value(self, game, depth, alphabeta=False, alpha=float("-inf"), beta=float("inf")):
        """
        This handles the min level of the game tree.

        Parameters
        ----------
        game - a Board object representing the current game state
        depth - the depth we have left to search
        alphabeta - bool, True if we are using alphabeta pruning
        alpha - the lower bound for alphabeta
        beta - the upper bound for alphabeta

        Returns
        -------
        float - the score for the current search branch
        tuple(int, int) - the best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if depth == 0:
            return self.score(game, self), (-1, -1)
        utility = float("inf")
        next_move = (-1, -1)
        for move in game.get_legal_moves(game.get_opponent(self)):
            this_score, _ = self.max_value(game.forecast_move(move), depth - 1, alphabeta, alpha, beta)
            # If we have found a better score, we update it.
            if utility > this_score:
                utility = this_score
                next_move = move
            if alphabeta:
                # We prune, as in we do not need to calculate further.
                if utility <= alpha:
                    break
                beta = min(beta, utility)
        return utility, next_move
