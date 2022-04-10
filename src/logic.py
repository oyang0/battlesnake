import random
from typing import List, Dict

from pickle import load
from copy import deepcopy

import numpy as np

"""
This file can be a nice home for your Battlesnake's logic and helper functions.

We have started this for you, and included some logic to remove your Battlesnake's 'neck'
from the list of possible moves!
"""


class NNUE:
    "An efficiently updatable neural network for evaluation"

    def __init__(self, ft_weight, ft_bias, l1_weight, l1_bias, l2_weight, l2_bias):
        self.ft_weight = ft_weight
        self.ft_bias = ft_bias
        self.l1_weight = l1_weight
        self.l1_bias = l1_bias
        self.l2_weight = l2_weight
        self.l2_bias = l2_bias

        self.accumulator = None
        self.refresh_accumulator([])

    def forward(self, features=None):
        accumulator = self._ft(features) if features else self.accumulator
        l1_x = self._relu(accumulator)
        l2_x = self._relu(self._l1(l1_x))

        return self._l2(l2_x)

    def refresh_accumulator(self, active_features):
        self.accumulator = self.ft_bias.copy()
        for active_feature in active_features:
            self.accumulator += self.ft_weight[:, active_feature]

    def update_accumulator(self, removed_features, added_features):
        for removed_feature in removed_features:
            self.accumulator -= self.ft_weight[:, removed_feature]
        for added_feature in added_features:
            self.accumulator += self.ft_weight[:, added_feature]

    def _ft(self, features):
        return self.ft_weight @ features + self.ft_bias

    def _l1(self, features):
        return self.l1_weight @ features + self.l1_bias

    def _l2(self, features):
        return self.l2_weight @ features + self.l2_bias

    def _relu(self, features):
        return np.maximum(0, features)


def _get_feature_mapping() -> dict:
    """
    return: The dictionary mapping features to indices
    """
    game_types = ("constrictor", "royale", "solo", "squad", "standard", "wrapped")
    board_sizes = (7, 11, 19)
    pieces = (
        'you_left',
        'you_right',
        'you_down',
        'you_up',
        'you_bottom',
        'you_head',
        'squad_left',
        'squad_right',
        'squad_down',
        'squad_up',
        'squad_bottom',
        'squad_head',
        'snake_left',
        'snake_right',
        'snake_down',
        'snake_up',
        'snake_bottom',
        'snake_head',
        'food',
        'hazard',
    )

    feature_mapping = {
        game_type: {
            board_size: {
                x: {y: {piece: None for piece in pieces} for y in range(board_size)}
                for x in range(board_size)
            }
            for board_size in board_sizes
        }
        for game_type in game_types
    }

    i = 0
    for game_type in game_types:
        for board_size in board_sizes:
            for x in range(board_size):
                for y in range(board_size):
                    for piece in pieces:
                        feature_mapping[game_type][board_size][x][y][piece] = i
                        i += 1

    return feature_mapping


def _get_piece_mapping() -> dict:
    """
    return: The dictionary mapping battlesnakes to pieces
    """
    piece_mapping = {
        battlesnake: {
            piece: f'{battlesnake}_{piece}'
            for piece in ['left', 'right', 'down', 'up', 'bottom', 'head']
        }
        for battlesnake in ['you', 'squad', 'snake']
    }

    return piece_mapping


games = {}
models = []
num_models = 0

fo = open("src/nnue.pth", "rb")
nnue = load(fo)
fo.close()
del fo

nnue = NNUE(
    nnue['ft.weight'],
    nnue['ft.bias'],
    nnue['l1.weight'],
    nnue['l1.bias'],
    nnue['l2.weight'],
    nnue['l2.bias'],
)
models.append(nnue)
num_models += 1

move_mapping = {0: "left", 1: "right", 2: "down", 3: "up"}
feature_mapping = _get_feature_mapping()
piece_mapping = _get_piece_mapping()


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


def choose_start(data: dict) -> None:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: None.

    Use the information in 'data' to initialize your next game. The 'data' variable can be interacted
    with as a Python Dictionary, and contains all of the information about the Battlesnake board
    for each move of the game.

    """
    if models:
        model = models.pop()
    elif num_models <= 32:
        model = deepcopy(nnue)
        num_models += 1
    else:
        model = None
    active_features = _refresh_state(data) if model else None
    if model:
        model.refresh_accumulator(active_features)

    if data["game"]["id"] not in games:
        games[data["game"]["id"]] = {}
    games[data["game"]["id"]][data["you"]["id"]] = {
        'model': model,
        'state': active_features,
    }


def _refresh_state(data: dict) -> list:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: The list of active features

    """
    active_features = set()

    mapping = feature_mapping[data["game"]["ruleset"]["name"]]
    mapping = mapping[data["board"]["height"]]

    for food in data["board"]["food"]:
        active_features.add(mapping[food["x"]][food["y"]]["food"])

    for hazard in data["board"]["hazards"]:
        active_features.add(mapping[hazard["x"]][hazard["y"]]["hazard"])

    for snake in data["board"]["snakes"]:
        if snake["id"] == data["you"]["id"]:
            pieces = piece_mapping["you"]
        elif (
            data["game"]["ruleset"]["name"] == "squad"
            and snake["squad"] == data["you"]["squad"]
        ):
            pieces = piece_mapping["squad"]
        else:
            pieces = piece_mapping["snake"]

        active_features.add(
            mapping[snake["head"]["x"]][snake["head"]["y"]][pieces["head"]]
        )

        for n, body in enumerate(snake["body"][1:]):
            previous_body = snake["body"][n]

            if previous_body["x"] < body["x"]:
                active_features.add(mapping[body["x"]][body["y"]][pieces["left"]])
            elif previous_body["x"] > body["x"]:
                active_features.add(mapping[body["x"]][body["y"]][pieces["right"]])
            elif previous_body["y"] < body["y"]:
                active_features.add(mapping[body["x"]][body["y"]][pieces["down"]])
            elif previous_body["y"] > body["y"]:
                active_features.add(mapping[body["x"]][body["y"]][pieces["up"]])
            else:
                active_features.add(mapping[body["x"]][body["y"]][pieces["bottom"]])

    active_features = [active_feature for active_feature in active_features]

    return active_features


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
    # print(f"~~~ Turn: {data['turn']}  Game Mode: {data['game']['ruleset']['name']} ~~~")
    # print(f"All board data this turn: {data}")
    # print(f"My Battlesnake this turn is: {my_snake}")
    # print(f"My Battlesnakes head this turn is: {my_head}")
    # print(f"My Battlesnakes body this turn is: {my_body}")

    possible_moves = ["up", "down", "left", "right"]

    # Step 0: Don't allow your Battlesnake to move back on it's own neck.
    possible_moves = _avoid_my_neck(my_body, possible_moves)

    # TODO: Step 1 - Don't hit walls.
    # Use information from `data` and `my_head` to not move beyond the game board.
    board = data['board']
    board_height = board['height']
    board_width = board['width']
    if data["game"]["ruleset"]["name"] != "wrapped":
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
    # move = random.choice(possible_moves) if possible_moves else "up"
    # TODO: Explore new strategies for picking a move that are better than random
    if possible_moves:
        model = games[data["game"]["id"]][data["you"]["id"]]["model"]
        if model:
            state = games[data["game"]["id"]][data["you"]["id"]]["state"]
            next_state = _refresh_state(data)
            removed_features, added_features = _update_state(state, next_state)
            games[data["game"]["id"]][data["you"]["id"]]["state"] = next_state
            model.update_accumulator(removed_features, added_features)
            sorted_moves = model.forward().argsort()[::-1]
            for sorted_move in sorted_moves:
                if move_mapping[sorted_move] in possible_moves:
                    move = move_mapping[sorted_move]
                    break
        else:
            move = random.choice(possible_moves)
    else:
        move = "up"

    print(
        f"{data['game']['id']} MOVE {data['turn']}: {move} picked from all valid options in {possible_moves}"
    )

    return move


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
    my_body_ = {(body["x"], body["y"]) for body in my_body}
    possible_moves_ = {possible_move for possible_move in possible_moves}

    if (
        my_head["x"] - 1,
        my_head["y"],
    ) in my_body_ and "left" in possible_moves_:  # my body is to the left of my head
        possible_moves.remove("left")
    if (
        my_head["x"] + 1,
        my_head["y"],
    ) in my_body_ and "right" in possible_moves_:  # my body is to the right of my head
        possible_moves.remove("right")
    if (
        my_head["x"],
        my_head["y"] - 1,
    ) in my_body_ and "down" in possible_moves_:  # my body is under my head
        possible_moves.remove("down")
    if (
        my_head["x"],
        my_head["y"] + 1,
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
    bodies = {
        (body["x"], body["y"])
        for snake in data["board"]["snakes"]
        for body in snake["body"]
        if snake["id"] != data["you"]["id"]
    }
    possible_moves_ = {possible_move for possible_move in possible_moves}

    if (
        my_head["x"] - 1,
        my_head["y"],
    ) in bodies and "left" in possible_moves_:  # others are to the left of my head
        possible_moves.remove("left")
    if (
        my_head["x"] + 1,
        my_head["y"],
    ) in bodies and "right" in possible_moves_:  # others are to the right of my head
        possible_moves.remove("right")
    if (
        my_head["x"],
        my_head["y"] - 1,
    ) in bodies and "down" in possible_moves_:  # others are under my head
        possible_moves.remove("down")
    if (
        my_head["x"],
        my_head["y"] + 1,
    ) in bodies and "up" in possible_moves_:  # others are over my head
        possible_moves.remove("up")

    return possible_moves


def _update_state(state: list, next_state: list) -> list:
    """
    state: List of new active features.
            e.g. [0, 10, 20]

    next_state: List of new active features.
            e.g. [0, 10, 20]

    return: The list of removed features and the list of added features

    """
    state_ = {feature for feature in state}
    next_state_ = {next_feature for next_feature in next_state_}

    removed_features = [feature for feature in state if feature not in next_state_]
    added_features = [
        next_feature for next_feature in next_state if next_feature not in state_
    ]

    return removed_features, added_features


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


def choose_end(data: dict) -> None:
    """
    data: Dictionary of all Game Board data as received from the Battlesnake Engine.
    For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

    return: None.

    Use the information in 'data' to delete your previous game. The 'data' variable can be interacted
    with as a Python Dictionary, and contains all of the information about the Battlesnake board
    for each move of the game.

    """
    model = games[data["game"]["id"]][data["you"]["id"]]["model"]
    models.append(model)
    if len(games[data["game"]["id"]]) > 1:
        del games[data["game"]["id"]][data["you"]["id"]]
    else:
        del games[data["game"]["id"]]
