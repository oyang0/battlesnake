import random
from typing import List, Dict

"""
This file can be a nice home for your Battlesnake's logic and helper functions.

We have started this for you, and included some logic to remove your Battlesnake's 'neck'
from the list of possible moves!
"""


def get_info() -> dict:
    """
    This controls your Battlesnake appearance and author permissions.
    For customization options, see https://docs.battlesnake.com/references/personalization

    TIP: If you open your Battlesnake URL in browser you should see this data.
    """
    return {
        "apiversion": "1",
        "author": "daedalean",  # TODO: Your Battlesnake Username
        "color": "#E80978",  # TODO: Personalize
        "head": "pixel",  # TODO: Personalize
        "tail": "pixel",  # TODO: Personalize
        "version": "0.0.1-beta",
    }


def choose_move(data: dict) -> str:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: A String, the single move to make. One of "up", "down", "left" or "right".

    Use the information in 'data' to decide your next move. The 'data' variable can be interacted
    with as a Python Dictionary, and contains all of the information about the Battlesnake board
    for each move of the game.

    """
    my_snake = data["you"]  # A dictionary describing your snake's position on the board
    my_head = my_snake["head"]  # A dictionary of coordinates like {"x": 0, "y": 0}
    my_body = my_snake[
        "body"
    ]  # A list of coordinate dictionaries like [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]

    # Uncomment the lines below to see what this data looks like in your output!
    print(f"~~~ Turn: {data['turn']}  Game Mode: {data['game']['ruleset']['name']} ~~~")
    print(f"All board data this turn: {data}")
    print(f"My Battlesnake this turn is: {my_snake}")
    print(f"My Battlesnakes head this turn is: {my_head}")
    print(f"My Battlesnakes body this turn is: {my_body}")

    possible_moves = ["up", "down", "left", "right"]

    # Step 0: Don't allow your Battlesnake to move back on it's own neck.
    possible_moves = _avoid_my_neck(my_body, possible_moves)

    # TODO: Step 1 - Don't hit walls.
    # Use information from `data` and `my_head` to not move beyond the game board.
    board = data['board']
    board_height = board['height']
    board_width = board['width']
    possible_moves = _avoid_hitting_walls(
        my_body, possible_moves, board_height, board_width
    )

    # TODO: Step 2 - Don't hit yourself.
    # Use information from `my_body` to avoid moves that would collide with yourself.
    possible_moves = _avoid_hitting_myself(my_body, possible_moves)

    # TODO: Step 3 - Don't collide with others.
    # Use information from `data` to prevent your Battlesnake from colliding with others.
    possible_moves = _avoid_colliding_others(my_body, possible_moves, data)

    # TODO: Step 4 - Find food.
    # Use information in `data` to seek out and find food.
    # food = data['board']['food']

    # Choose a random direction from the remaining possible_moves to move in, and then return that move
    move = random.choice(possible_moves)
    # TODO: Explore new strategies for picking a move that are better than random

    print(
        f"{data['game']['id']} MOVE {data['turn']}: {move} picked from all valid options in {possible_moves}"
    )

    return move


def choose_shout(data: dict, move: str) -> str:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    move: A String, the single move to make. One of "up", "down", "left" or "right".

    return: A String, the single shout to make.

    Use the information in 'data' and 'move' to decide your next shout. The 'data' variable can be interacted
    with as a Python Dictionary, and contains all of the information about the Battlesnake board
    for each move of the game.

    """
    shouts = [
        "why are we shouting??",
        "I'm not really sure...",
        f"I guess I'll go {move} then.",
    ]
    shout = random.choice(shouts)
    return shout


def _avoid_my_neck(my_body: dict, possible_moves: List[str]) -> List[str]:
    """
    my_body: List of dictionaries of x/y coordinates for every segment of a Battlesnake.
            e.g. [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]
    possible_moves: List of strings. Moves to pick from.
            e.g. ["up", "down", "left", "right"]

    return: The list of remaining possible_moves, with the 'neck' direction removed
    """
    my_head = my_body[0]  # The first body coordinate is always the head
    my_neck = my_body[1]  # The segment of body right after the head is the 'neck'
    possible_moves_ = {possible_move for possible_move in possible_moves}

    if (
        my_neck["x"] < my_head["x"] and "left" in possible_moves_
    ):  # my neck is left of my head
        possible_moves.remove("left")
    elif (
        my_neck["x"] > my_head["x"] and "right" in possible_moves_
    ):  # my neck is right of my head
        possible_moves.remove("right")
    elif (
        my_neck["y"] < my_head["y"] and "down" in possible_moves_
    ):  # my neck is below my head
        possible_moves.remove("down")
    elif (
        my_neck["y"] > my_head["y"] and "up" in possible_moves_
    ):  # my neck is above my head
        possible_moves.remove("up")

    return possible_moves


def _avoid_hitting_walls(
    my_body: dict, possible_moves: List[str], board_height: int, board_width: int
) -> List[str]:
    """
    my_body: List of dictionaries of x/y coordinates for every segment of a Battlesnake.
            e.g. [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]
    possible_moves: List of strings. Moves to pick from.
            e.g. ["up", "down", "left", "right"]

    return: The list of remaining possible_moves, with 'wall' directions removed
    """
    my_head = my_body[0]  # The first body coordinate is always the head
    possible_moves_ = {possible_move for possible_move in possible_moves}

    if 0 == my_head["x"] and "left" in possible_moves_:  # my head is at the left wall
        possible_moves.remove("left")
    if (
        board_width - 1 == my_head["x"] and "right" in possible_moves_
    ):  # my head is at the right wall
        possible_moves.remove("right")
    if 0 == my_head["y"] and "down" in possible_moves_:  # my head is at the bottom wall
        possible_moves.remove("down")
    if (
        board_height - 1 == my_head["y"] and "up" in possible_moves_
    ):  # my head is at the top wall
        possible_moves.remove("up")

    return possible_moves


def _avoid_hitting_myself(my_body: dict, possible_moves: List[str]) -> List[str]:
    """
    my_body: List of dictionaries of x/y coordinates for every segment of a Battlesnake.
            e.g. [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]
    possible_moves: List of strings. Moves to pick from.
            e.g. ["up", "down", "left", "right"]

    return: The list of remaining possible_moves, with 'body' directions removed
    """
    my_head = my_body[0]  # The first body coordinate is always the head
    my_body_ = {tuple(body.items()) for body in my_body}
    possible_moves_ = {possible_move for possible_move in possible_moves}

    if (
        ("x", my_head["x"] - 1),
        ("y", my_head["y"]),
    ) in my_body_ and "left" in possible_moves_:  # my body is to the left of my head
        possible_moves.remove("left")
    if (
        ("x", my_head["x"] + 1),
        ("y", my_head["y"]),
    ) in my_body_ and "right" in possible_moves_:  # my body is to the right of my head
        possible_moves.remove("right")
    if (
        ("x", my_head["x"]),
        ("y", my_head["y"] - 1),
    ) in my_body_ and "down" in possible_moves_:  # my body is under my head
        possible_moves.remove("down")
    if (
        ("x", my_head["x"]),
        ("y", my_head["y"] + 1),
    ) in my_body_ and "up" in possible_moves_:  # my body is over my head
        possible_moves.remove("up")

    return possible_moves


def _avoid_colliding_others(
    my_body: dict, possible_moves: List[str], data: dict
) -> List[str]:
    """
    my_body: List of dictionaries of x/y coordinates for every segment of a Battlesnake.
            e.g. [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]
    possible_moves: List of strings. Moves to pick from.
            e.g. ["up", "down", "left", "right"]
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
            For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: The list of remaining possible_moves, with 'others' directions removed
    """
    my_head = my_body[0]  # The first body coordinate is always the head
    body = {
        tuple(body.items())
        for snake in data["board"]["snakes"]
        for body in snake["body"]
    }
    possible_moves_ = {possible_move for possible_move in possible_moves}

    if (
        ("x", my_head["x"] - 1),
        ("y", my_head["y"]),
    ) in body and "left" in possible_moves_:  # others are to the left of my head
        possible_moves.remove("left")
    if (
        ("x", my_head["x"] + 1),
        ("y", my_head["y"]),
    ) in body and "right" in possible_moves_:  # others are to the right of my head
        possible_moves.remove("right")
    if (
        ("x", my_head["x"]),
        ("y", my_head["y"] - 1),
    ) in body and "down" in possible_moves_:  # others are under my head
        possible_moves.remove("down")
    if (
        ("x", my_head["x"]),
        ("y", my_head["y"] + 1),
    ) in body and "up" in possible_moves_:  # others are over my head
        possible_moves.remove("up")

    return possible_moves
