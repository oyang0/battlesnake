import random

from pickle import load
from copy import deepcopy

import tools


class Logic:
    """
    This file can be a nice home for your Battlesnake's logic and helper functions.

    We have started this for you, and included some logic to remove your Battlesnake's 'neck'
    from the list of possible moves!
    """

    def __init__(self, model_file="src/model.pth", max_models=4):
        with open(model_file, "rb") as model:
            self.model = load(model)

        self.feature_mapping = self._get_feature_mapping()
        self.move_mapping = {0: "left", 1: "right", 2: "down", 3: "up"}

        self.models = {}
        self.active_features = {}
        self.recycle_bin = [self.model]

        self.game_types = {
            "constrictor",
            "royale",
            "solo",
            "squad",
            "standard",
            "wrapped",
        }
        self.board_sizes = {7, 11, 19}

        self.max_models = max_models

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
        game_type = tools.get_game_type(data)
        board_size = tools.get_board_size(data)

        if game_type in self.game_types and board_size in self.board_sizes:

            if self.recycle_bin():
                game_id = tools.get_game_id(data)
                your_id = tools.get_your_id(data)

                model = self.recycle_bin.pop()
                active_features = self._get_active_features(data)
                model.refresh_accumulator(active_features)

                self.models[(game_id, your_id)] = model
                self.active_features[(game_id, your_id)] = active_features

            elif len(self.models) + len(self.recycle_bin) <= self.max_models:
                game_id = tools.get_game_id(data)
                your_id = tools.get_your_id(data)

                model = deepcopy(self.model)
                active_features = self._get_active_features(data)
                model.refresh_accumulator(active_features)

                self.models[(game_id, your_id)] = model
                self.active_features[(game_id, your_id)] = active_features

    def choose_move(self, data):
        """
        data: Dictionary of all Game Board data as received from the Battlesnake Engine.
        For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request

        return: A String, the single move to make. One of "up", "down", "left" or "right".

        Use the information in 'data' to decide your next move. The 'data' variable can be interacted
        with as a Python Dictionary, and contains all of the information about the Battlesnake board
        for each move of the game.

        """
        my_snake = data[
            "you"
        ]  # A dictionary describing your snake's position on the board
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
        possible_moves = self._avoid_my_neck(my_body, possible_moves)

        # TODO: Step 1 - Don't hit walls.
        # Use information from `data` and `my_head` to not move beyond the game board.
        board = data['board']
        board_height = board['height']
        board_width = board['width']
        possible_moves = self._avoid_hitting_walls(
            my_body, possible_moves, board_height, board_width, data
        )

        # TODO: Step 2 - Don't hit yourself.
        # Use information from `my_body` to avoid moves that would collide with yourself.
        possible_moves = self._avoid_hitting_myself(my_body, possible_moves)

        # TODO: Step 3 - Don't collide with others.
        # Use information from `data` to prevent your Battlesnake from colliding with others.
        possible_moves = self._avoid_colliding_others(my_body, possible_moves, data)

        # TODO: Step 4 - Find food.
        # Use information in `data` to seek out and find food.
        # food = data['board']['food']

        # Choose a random direction from the remaining possible_moves to move in, and then return that move
        # move = random.choice(possible_moves) if possible_moves else "up"
        # TODO: Explore new strategies for picking a move that are better than random
        if possible_moves:
            game_id = tools.get_game_id(data)
            your_id = tools.get_your_id(data)

            if (game_id, your_id) in self.models:
                previous_features = self.active_features[(game_id, your_id)]
                next_features = self._get_active_features(data)
                removed_features, added_features = self._get_removed_and_added_features(
                    previous_features, next_features
                )

                model = self.models[(game_id, your_id)]
                model.update_accumulator(removed_features, added_features)
                sorted_moves = model.forward().argsort()[::-1]
                
                self.active_features[(game_id, your_id)] = next_features

                for sorted_move in sorted_moves:
                    mapped_move = self.move_mapping[sorted_move]

                    if mapped_move in possible_moves:
                        move = mapped_move
                        break
            else:
                move = random.choice(possible_moves)
        else:
            move = "up"

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
        shout = random.choice(shouts)
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
        game_id = tools.get_game_id(data)
        your_id = tools.get_your_id(data)

        if (game_id, your_id) in self.models:
            model = self.models[(game_id, your_id)]
            self.recycle_bin.append(model)

            del self.models[(game_id, your_id)]
            del self.active_features[(game_id, your_id)]

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

    def _avoid_hitting_walls(
        self,
        my_body,
        possible_moves,
        board_height,
        board_width,
        data,
    ):
        """
        my_body: List of dictionaries of x/y coordinates for every segment of a Battlesnake.
                e.g. [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]
        possible_moves: List of strings. Moves to pick from.
                e.g. ["up", "down", "left", "right"]

        return: The list of remaining possible_moves, with 'wall' directions removed
        """
        my_head = my_body[0]  # The first body coordinate is always the head

        game_type = tools.get_game_type(data)

        if game_type != "wrapped":
            possible_move_set = tools.get_possible_move_set(possible_moves)
            adjacent_squares_and_moves = tools.get_adjacent_squares_and_moves(my_head)

            for adjacent_square, move in adjacent_squares_and_moves:
                out_of_bounds = tools.is_out_of_bounds(
                    adjacent_square, board_height, board_width
                )
                if out_of_bounds and move in possible_move_set:
                    possible_moves.remove(move)

        return possible_moves

    def _avoid_hitting_myself(self, my_body, possible_moves):
        """
        my_body: List of dictionaries of x/y coordinates for every segment of a Battlesnake.
                e.g. [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 2, "y": 0}]
        possible_moves: List of strings. Moves to pick from.
                e.g. ["up", "down", "left", "right"]

        return: The list of remaining possible_moves, with 'body' directions removed
        """
        my_head = my_body[0]  # The first body coordinate is always the head
        body_set_except_tail = tools.get_body_set_except_tail(my_body)
        possible_move_set = tools.get_possible_move_set(possible_moves)

        adjacent_squares_and_moves = tools.get_adjacent_squares_and_moves(my_head)

        for adjacent_square, move in adjacent_squares_and_moves:
            if adjacent_square in body_set_except_tail and move in possible_move_set:
                possible_moves.remove(move)

        return possible_moves

    def _avoid_colliding_others(self, my_body, possible_moves, data):
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
        other_body_set_except_squadmates_and_tails = (
            tools.get_other_body_set_except_squadmates_and_tails(data)
        )
        possible_move_set = {possible_move for possible_move in possible_moves}

        adjacent_squares_and_moves = tools.get_adjacent_squares_and_moves(my_head)

        for adjacent_square, move in adjacent_squares_and_moves:
            if (
                adjacent_square in other_body_set_except_squadmates_and_tails
                and move in possible_move_set
            ):
                possible_moves.remove(move)

        return possible_moves

    def _get_active_features(self, data):
        """
        data: Dictionary of all Game Board data as received from the Battlesnake Engine.
                For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request
        return: The list of active features
        """
        active_features = set()

        your_id = tools.get_your_id(data)
        your_squad = tools.get_your_squad(data)
        game_type = tools.get_game_type(data)
        board_size = tools.get_board_size(data)
        foods = tools.get_food(data)
        hazards = tools.get_hazards(data)
        snakes = tools.get_snakes(data)

        for food in foods:
            active_feature = (game_type, board_size, food["x"], food["y"], "food")
            active_features.add(self.feature_mapping[active_feature])

        for hazard in hazards:
            active_feature = (game_type, board_size, hazard["x"], hazard["y"], "hazard")
            active_features.add(self.feature_mapping[active_feature])

        for snake in snakes:
            if snake["id"] == your_id:
                player = "you"
            elif game_type == "squad" and snake["squad"] == your_squad:
                player = "squad"
            else:
                player = "snake"

            head = snake["head"]
            x = head["x"]
            y = head["y"]
            health = snake["health"]
            active_feature = (game_type, board_size, x, y, player, health)
            active_features.add(self.feature_mapping[active_feature])

            for previous_body, body in zip(snake["body"][:-1], snake["body"][1:]):
                previous_x = previous_body["x"]
                previous_y = previous_body["y"]
                x = body["x"]
                y = body["y"]

                if previous_x < x:
                    active_feature = (game_type, board_size, x, y, player, "left")
                elif previous_x > x:
                    active_feature = (game_type, board_size, x, y, player, "right")
                elif previous_y < y:
                    active_feature = (game_type, board_size, x, y, player, "down")
                elif previous_y > y:
                    active_feature = (game_type, board_size, x, y, player, "up")
                else:
                    active_feature = (game_type, board_size, x, y, player, "bottom")

                active_features.add(self.feature_mapping[active_feature])

        active_features = tuple(active_features)

        return active_features

    def _get_removed_and_added_features(self, previous_features, next_features):
        """
        previous_features: List of previous features.
                e.g. [0, 10, 20]
        next_features: List of next features.
                e.g. [0, 10, 20]
        return: The list of removed features and the list of added features
        """
        previous_feature_set = {
            previous_feature for previous_feature in previous_features
        }
        next_feature_set = {active_feature for active_feature in next_features}

        removed_features = [
            previous_feature
            for previous_feature in previous_features
            if previous_feature not in next_feature_set
        ]
        added_features = [
            active_feature
            for active_feature in next_features
            if active_feature not in previous_feature_set
        ]

        return removed_features, added_features

    def _get_feature_mapping(self):
        """
        return: The dictionary of mapping features to indices
        """
        game_types = ["constrictor", "royale", "solo", "squad", "standard", "wrapped"]
        board_sizes = [7, 11, 19]
        players = ["you", "squad", "snake", "food", "hazard"]
        players_ = {"you", "squad", "snake"}
        pieces = ["left", "right", "down", "up", "bottom", "head"]
        pieces_ = {"head"}
        healths = range(1, 101)

        feature_mapping = {}
        index = 0
        for game_type in game_types:
            for board_size in board_sizes:
                for x in range(board_size):
                    for y in range(board_size):
                        for player in players:
                            if player in players_:
                                for piece in pieces:
                                    if piece in pieces_:
                                        for health in healths:
                                            feature_mapping[
                                                (
                                                    game_type,
                                                    board_size,
                                                    x,
                                                    y,
                                                    player,
                                                    piece,
                                                    health,
                                                )
                                            ] = index
                                            index += 1
                                    else:
                                        feature_mapping[
                                            (game_type, board_size, x, y, player, piece)
                                        ] = index
                                        index += 1
                            else:
                                feature_mapping[
                                    (game_type, board_size, x, y, player)
                                ] = index
                                index += 1

        return feature_mapping


if __name__ == "__main__":
    logic = Logic()
