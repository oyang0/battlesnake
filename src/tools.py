from typing import List, Tuple, Dict, Set

"""
Battlesnake helper functions.
"""


def get_adjacent_squares(square: dict) -> Tuple[tuple]:
    """
    square: Dictionary of x/y coordinates.
            e.g. {"x": 0, "y": 0}

    return: A tuple of adjacent squares

    """
    x = square["x"]
    y = square["y"]

    adjacent_squares = ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))

    return adjacent_squares


def get_adjacent_squares_and_moves(square: dict) -> Tuple[tuple]:
    """
    square: Dictionary of x/y coordinates.
            e.g. {"x": 0, "y": 0}

    return: A tuple of (adjacent square, move) tuples

    """
    x = square["x"]
    y = square["y"]

    adjacent_squares_and_moves = (
        ((x - 1, y), "left"),
        ((x + 1, y), "right"),
        ((x, y - 1), "down"),
        ((x, y + 1), "up"),
    )

    return adjacent_squares_and_moves


def is_out_of_bounds(square: tuple, board_height: int, board_width: int) -> bool:
    """
    square: Dictionary of x/y coordinates.
            e.g. {"x": 0, "y": 0}

    board_size: An Integer, the board size.
            e.g. 11

    return: A Boolean, if the square is out of bounds.

    """
    x = square[0]
    y = square[1]

    out_of_bounds = x < 0 or x >= board_width or y < 0 or y >= board_height

    return out_of_bounds


def get_possible_move_set(possible_moves: list) -> Set[str]:
    """
    possible_moves: List of strings. Moves to pick from.
            e.g. ["up", "down", "left", "right"]

    return: A set of possible moves

    """
    possible_move_set = {possible_move for possible_move in possible_moves}
    return possible_move_set


def get_body_set_except_tail(body: list) -> Set[tuple]:
    """
    body: List of x/y coordinates for every segment of a Battlesnake.
            e.g. [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]

    return: A set of x/y coordinates for every segment of a Battlesnake except the tail

    """
    body_set_except_tail = {(segment["x"], segment["y"]) for segment in body[:-1]}
    return body_set_except_tail


def get_other_body_set_except_squadmates_and_tails(data: dict) -> Set[tuple]:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A set of x/y coordinates for every segment of other Battlesnakes except squadmates and tails

    """
    your_id = get_your_id(data)
    your_squad = get_your_squad(data)
    game_type = get_game_type(data)
    snakes = get_snakes(data)

    other_body_set_except_squadmates_and_tails = {
        (body["x"], body["y"])
        for snake in snakes
        if snake["id"] != your_id
        and (game_type != "squad" or snake["squad"] != your_squad)
        for body in snake["body"][:-1]
    }

    return other_body_set_except_squadmates_and_tails


def get_game_id(data: dict) -> str:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A String, the game id.

    """
    return data["game"]["id"]


def get_game_type(data: dict) -> str:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A String, the game type.

    """
    return data["game"]["ruleset"]["name"]


def get_board_size(data: dict) -> int:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: An Integer, the board size.

    """
    return data["board"]["height"]


def get_foods(data: dict) -> List[dict]:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: The list of foods

    """
    return data["board"]["food"]


def get_hazards(data: dict) -> List[dict]:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: The list of hazards

    """
    return data["board"]["hazards"]


def get_snakes(data: dict) -> List[dict]:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: The list of snakes

    """
    return data["board"]["snakes"]


def get_snake_id(snake: dict) -> str:
    """
    snake: Dictionary of a snake's data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A String, a snake's id.

    """
    return snake["id"]


def get_snake_head(snake: dict) -> Dict[str, int]:
    """
    snake: Dictionary of a snake's data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A dictionary of x/y coordinates for the head of a snake

    """
    return snake["head"]


def get_snake_length(snake: dict) -> int:
    """
    snake: Dictionary of a snake's data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: An Integer, a snake's length.

    """
    return snake["length"]


def get_your_id(data: dict) -> str:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A String, your id.

    """
    return data["you"]["id"]


def get_your_length(data: dict) -> int:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: An Integer, your length.

    """
    return data["you"]["length"]


def get_your_squad(data: dict) -> str:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A String, your squad.

    """
    return data["you"]["squad"]
