from random import choice

from copy import copy

from logics.bad_moves import BadMoves
from logics.chaise_tail import ChaiseTail
from logics.eat import Eat
from logics.kill import Kill
from logics.orthogonal_distances import OrthogonalDistances
from logics.path_distances import PathDistances
from logics.increase_board_control import IncreaseBoardControl
from logics.surround import Surround

from utils.game_state import GameState
from utils.snake import Snake
from utils.vector import Vector, up, down, left, right, noop, directions

from src.floodfill import is_coords_open, calc_neighbors, calc_open_space
from src.pathfinding import calc_possible_moves, calc_next_move
from src.pathfinding import BattlesnakeAStarPathfinder
from src.targeting import calc_targets
from src.util import calc_manhattan_distance


class Logic:
    """
    This file can be a nice home for your Battlesnake's logic and helper functions.

    We have started this for you, and included some logic to remove your Battlesnake's 'neck'
    from the list of possible moves!
    """

    def __init__(self, model):
        self.model = model

        self.feature_mapping = self._get_feature_mapping()
        self.move_mapping = {0: "left", 1: "right", 2: "down", 3: "up"}

        self.models = {}
        self.features = {}

    def get_info(self):
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

    def choose_start(self, data):
        """
        data: Dictionary of all Game Board data as received from the Battlesnake Engine.
        For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

        return: None.

        Use the information in 'data' to initialize your next game. The 'data' variable can be interacted
        with as a Python Dictionary, and contains all of the information about the Battlesnake board
        for each move of the game.

        """
        len_snakes = len(data["board"]["snakes"])
        board_size = data["board"]["height"]
        game_type = data["game"]["ruleset"]["name"]

        if len_snakes == 2 and board_size == 11 and game_type == "standard":
            game_id = data["game"]["id"]
            my_id = data["you"]["id"]

            model = copy(self.model)
            active_features = self._get_active_features(data)
            model.refresh_accumulator(active_features)

            self.models[(game_id, my_id)] = model
            self.features[(game_id, my_id)] = active_features

    def choose_move(self, data):
        """
        data: Dictionary of all Game Board data as received from the Battlesnake Engine.
        For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

        return: A String, the single move to make. One of "up", "down", "left" or "right".

        Use the information in 'data' to decide your next move. The 'data' variable can be interacted
        with as a Python Dictionary, and contains all of the information about the Battlesnake board
        for each move of the game.

        """
        # my_snake = data["you"]  # A dictionary describing your snake's position on the board
        # my_head = my_snake["head"]  # A dictionary of coordinates like {"x": 0, "y": 0}
        # my_body = my_snake["body"]  # A list of coordinate dictionaries like [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]

        # Uncomment the lines below to see what this data looks like in your output!
        # print(f"~~~ Turn: {data['turn']}  Game Mode: {data['game']['ruleset']['name']} ~~~")
        # print(f"All board data this turn: {data}")
        # print(f"My Battlesnake this turn is: {my_snake}")
        # print(f"My Battlesnakes head this turn is: {my_head}")
        # print(f"My Battlesnakes body this turn is: {my_body}")

        # possible_moves = ["up", "down", "left", "right"]

        # Step 0: Don't allow your Battlesnake to move back on it's own neck.
        # possible_moves = self._avoid_my_neck(my_body, possible_moves)

        # TODO: Step 1 - Don't hit walls.
        # Use information from `data` and `my_head` to not move beyond the game board.
        # board = data['board']
        # board_height = board['height']
        # board_width = board['width']

        # TODO: Step 2 - Don't hit yourself.
        # Use information from `my_body` to avoid moves that would collide with yourself.

        # TODO: Step 3 - Don't collide with others.
        # Use information from `data` to prevent your Battlesnake from colliding with others.
        possible_moves = calc_possible_moves(data)

        # TODO: Step 4 - Find food.
        # Use information in `data` to seek out and find food.
        # food = data['board']['food']

        # Choose a random direction from the remaining possible_moves to move in, and then return that move
        # move = choice(possible_moves) if possible_moves else "up"
        # TODO: Explore new strategies for picking a move that are better than random
        if possible_moves:
            if len(possible_moves) > 1:
                game_id = data["game"]["id"]
                my_id = data["you"]["id"]

                if (game_id, my_id) in self.models:
                    previous_features = self.features[(game_id, my_id)]
                    next_features = self._get_active_features(data)
                    removed_features = self._get_removed_features(
                        previous_features, next_features
                    )
                    added_features = self._get_added_features(
                        previous_features, next_features
                    )

                    model = self.models[(game_id, my_id)]
                    model.update_accumulator(removed_features, added_features)
                    sorted_moves = model.forward().argsort()[::-1]

                    self.features[(game_id, my_id)] = next_features

                    for sorted_move in sorted_moves:
                        mapped_move = self.move_mapping[sorted_move]

                        if mapped_move in possible_moves:
                            move = mapped_move
                            break
                else:
                    move = choice(possible_moves)
            else:
                move = possible_moves[0]
        else:
            move = choice(["up", "down", "left", "right"])

        print(
            f"{data['game']['id']} MOVE {data['turn']}: {move} picked from all valid options in {possible_moves}"
        )

        return move

    def choose_shout(self, data, move):
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
        shout = choice(shouts)
        return shout

    def choose_end(self, data):
        """
        data: Dictionary of all Game Board data as received from the Battlesnake Engine.
        For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

        return: None.

        Use the information in 'data' to delete your previous game. The 'data' variable can be interacted
        with as a Python Dictionary, and contains all of the information about the Battlesnake board
        for each move of the game.

        """
        game_id = data["game"]["id"]
        my_id = data["you"]["id"]

        if (game_id, my_id) in self.models:
            del self.models[(game_id, my_id)]
            del self.features[(game_id, my_id)]

    def _avoid_my_neck(self, my_body, possible_moves):
        """
        my_body: List of dictionaries of x/y coordinates for every segment of a Battlesnake.
                e.g. [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]
        possible_moves: List of strings. Moves to pick from.
                e.g. ["up", "down", "left", "right"]
        return: The list of remaining possible_moves, with the 'neck' direction removed
        """
        my_head = my_body[0]  # The first body coordinate is always the head
        my_neck = my_body[1]  # The segment of body right after the head is the 'neck'

        if my_neck["x"] < my_head["x"]:  # my neck is left of my head
            possible_moves.remove("left")
        elif my_neck["x"] > my_head["x"]:  # my neck is right of my head
            possible_moves.remove("right")
        elif my_neck["y"] < my_head["y"]:  # my neck is below my head
            possible_moves.remove("down")
        elif my_neck["y"] > my_head["y"]:  # my neck is above my head
            possible_moves.remove("up")

        return possible_moves

    def _get_active_features(self, data):
        """
        data: Dictionary of all Game Board data as received from the Battlesnake Engine.
                For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request
        return: The list of active features
        """
        my_id = data["you"]["id"]
        foods = data["board"]["food"]
        snakes = data["board"]["snakes"]

        active_features = set()

        for food in foods:
            square = (food["x"], food["y"])
            active_features.add(self.feature_mapping[(square, "food")])

        for snake in snakes:
            color = "you" if snake["id"] == my_id else "snake"
            health = ("health", snake["health"])
            length = ("length", snake["length"])

            head = snake["head"]
            square = (head["x"], head["y"])
            active_features.add(self.feature_mapping[(square, health)])
            active_features.add(self.feature_mapping[(square, "head")])
            active_features.add(self.feature_mapping[(square, length)])
            active_features.add(self.feature_mapping[(square, color)])

            for body in snake["body"][1:]:
                square = (body["x"], body["y"])
                active_features.add(self.feature_mapping[(square, health)])
                active_features.add(self.feature_mapping[(square, "body")])
                active_features.add(self.feature_mapping[(square, length)])
                active_features.add(self.feature_mapping[(square, color)])

            for previous_body, body, next_body in zip(
                snake["body"][:-2], snake["body"][1:-1], snake["body"][2:]
            ):
                square = (body["x"], body["y"])

                if previous_body["x"] < body["x"]:
                    active_features.add(
                        self.feature_mapping[(square, ("previous", "left"))]
                    )
                elif previous_body["x"] > body["x"]:
                    active_features.add(
                        self.feature_mapping[(square, ("previous", "right"))]
                    )
                elif previous_body["y"] < body["y"]:
                    active_features.add(
                        self.feature_mapping[(square, ("previous", "down"))]
                    )
                elif previous_body["y"] > body["y"]:
                    active_features.add(
                        self.feature_mapping[(square, ("previous", "up"))]
                    )
                else:
                    active_features.add(
                        self.feature_mapping[(square, ("previous", "noop"))]
                    )

                if next_body["x"] < body["x"]:
                    active_features.add(
                        self.feature_mapping[(square, ("next", "left"))]
                    )
                elif next_body["x"] > body["x"]:
                    active_features.add(
                        self.feature_mapping[(square, ("next", "right"))]
                    )
                elif next_body["y"] < body["y"]:
                    active_features.add(
                        self.feature_mapping[(square, ("next", "down"))]
                    )
                elif next_body["y"] > body["y"]:
                    active_features.add(self.feature_mapping[(square, ("next", "up"))])
                else:
                    active_features.add(
                        self.feature_mapping[(square, ("next", "noop"))]
                    )

        active_features = tuple(active_features)

        return active_features

    def _get_removed_features(self, previous_features, next_features):
        """
        previous_features: List of previous features.
                e.g. [0, 10, 20]
        next_features: List of next features.
                e.g. [0, 10, 20]
        return: The list of removed features
        """
        removed_features = [
            previous_feature
            for previous_feature in previous_features
            if previous_feature not in next_features
        ]

        return removed_features

    def _get_added_features(self, previous_features, next_features):
        """
        previous_features: List of previous features.
                e.g. [0, 10, 20]
        next_features: List of next features.
                e.g. [0, 10, 20]
        return: The list of added features
        """
        added_features = [
            active_feature
            for active_feature in next_features
            if active_feature not in previous_features
        ]

        return added_features

    def _get_feature_mapping(self):
        """
        return: The dictionary of mapping features to indices
        """
        healths = range(1, 100 + 1)
        piece_types = ["body", "head"]
        lengths = range(3, 11 * 11 + 1)
        directions = ["up", "down", "left", "right", "noop"]
        colors = ["you", "snake"]

        feature_mapping = {}
        index = 0
        for x in range(11):
            for y in range(11):

                for health in healths:
                    feature_mapping[((x, y), ("health", health))] = index
                    index += 1

                for piece_type in piece_types:
                    feature_mapping[((x, y), piece_type)] = index
                    index += 1

                for length in lengths:
                    feature_mapping[((x, y), ("length", length))] = index
                    index += 1

                for direction in directions:
                    feature_mapping[((x, y), ("next", direction))] = index
                    index += 1

                for direction in directions:
                    feature_mapping[((x, y), ("previous", direction))] = index
                    index += 1

                for color in colors:
                    feature_mapping[((x, y), color)] = index
                    index += 1

                feature_mapping[((x, y), "food")] = index
                index += 1

        return feature_mapping


if __name__ == "__main__":
    logic = Logic()
