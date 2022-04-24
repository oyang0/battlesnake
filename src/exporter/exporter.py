import torch

class Exporter:
    "Generate state–action–reward–state–action quintuples from Battlesnake games and snakes"

    def __init__(self, prefix):
        self.prefix = prefix

        self.games = {}
        self.turns = {}

        self.feature_mapping = self._get_feature_mapping()

    def start(self, data):
        game_id = data["game"]["id"]
        my_id = data["you"]["id"]
        turn = data["turn"]

        self.games[(game_id, my_id)] = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "next_actions": [],
        }

        self.turns[(game_id, my_id)] = turn

    def move(self, data, move):
        game_id = data["game"]["id"]
        my_id = data["you"]["id"]
        turn = data["turn"]

        state = self._get_state(data)
        action = self._get_action(move)
        reward = self._get_reward(data)

        if self.games[(game_id, my_id)]["states"]:
            self.games[(game_id, my_id)]["rewards"].append(reward)
            self.games[(game_id, my_id)]["next_states"].append(state)
            self.games[(game_id, my_id)]["next_actions"].append(action)

        self.games[(game_id, my_id)]["states"].append(state)
        self.games[(game_id, my_id)]["actions"].append(action)

        self.turns[(game_id, my_id)] = turn

    def end(self, data):
        game_id = data["game"]["id"]
        my_id = data["you"]["id"]
        turn = data["turn"]

        if self.games[(game_id, my_id)]["states"]:
            state = self._get_state(data)
            action = self._get_action(None)
            reward = self._get_reward(data)

            self.games[(game_id, my_id)]["rewards"].append(reward)
            self.games[(game_id, my_id)]["next_states"].append(state)
            self.games[(game_id, my_id)]["next_actions"].append(action)

        torch.save(self.games[(game_id, my_id)], f"{self.prefix}/{game_id}_{my_id}.pt")

        del self.games[(game_id, my_id)]
        del self.turns[(game_id, my_id)]

    def _get_state(self, data):
        my_id = data["you"]["id"]
        foods = data["board"]["food"]
        snakes = data["board"]["snakes"]

        features = {
            self.feature_mapping[((food["x"], food["y"]), "food")] for food in foods
        }

        for snake in snakes:
            color = "you" if snake["id"] == my_id else "snake"
            health = ("health", snake["health"])
            length = ("length", snake["length"])

            head = snake["head"]
            square = (head["x"], head["y"])
            features.add(self.feature_mapping[(square, health)])
            features.add(self.feature_mapping[(square, "head")])
            features.add(self.feature_mapping[(square, length)])
            features.add(self.feature_mapping[(square, color)])

            for body in snake["body"][1:]:
                square = (body["x"], body["y"])
                features.add(self.feature_mapping[(square, health)])
                features.add(self.feature_mapping[(square, "body")])
                features.add(self.feature_mapping[(square, length)])
                features.add(self.feature_mapping[(square, color)])

            for previous_body, body, next_body in zip(
                snake["body"][:-2], snake["body"][1:-1], snake["body"][2:]
            ):
                square = (body["x"], body["y"])

                if previous_body["x"] < body["x"]:
                    features.add(self.feature_mapping[(square, ("previous", "left"))])
                elif previous_body["x"] > body["x"]:
                    features.add(self.feature_mapping[(square, ("previous", "right"))])
                elif previous_body["y"] < body["y"]:
                    features.add(self.feature_mapping[(square, ("previous", "down"))])
                elif previous_body["y"] > body["y"]:
                    features.add(self.feature_mapping[(square, ("previous", "up"))])
                else:
                    features.add(self.feature_mapping[(square, ("previous", "noop"))])

                if next_body["x"] < body["x"]:
                    features.add(self.feature_mapping[(square, ("next", "left"))])
                elif next_body["x"] > body["x"]:
                    features.add(self.feature_mapping[(square, ("next", "right"))])
                elif next_body["y"] < body["y"]:
                    features.add(self.feature_mapping[(square, ("next", "down"))])
                elif next_body["y"] > body["y"]:
                    features.add(self.feature_mapping[(square, ("next", "up"))])
                else:
                    features.add(self.feature_mapping[(square, ("next", "noop"))])

        if self._is_final_state(data):
            value = float('nan')
        else:
            value = 1.0

        state = torch.sparse_coo_tensor(
            [[feature for feature in features]],
            [value for _ in range(len(features))],
            (len(self.feature_mapping),),
        )

        return state

    def _get_action(self, move):
        if move == "left":
            action = torch.tensor([0])
        elif move == "right":
            action = torch.tensor([1])
        elif move == "down":
            action = torch.tensor([2])
        elif move == "up":
            action = torch.tensor([3])
        elif not move:
            action = torch.tensor([-9223372036854775808])  # NaN to long is -9223372036854775808
        else:
            raise Exception("no action returned")
    
        return action
    
    
    def _get_reward(self, data):
        """
        data: Dictionary of all Game Board data as received from the Battlesnake Engine.
        For a full example of 'data', see https://docs.battlesnake.com/references/api/sample-move-request
    
        return: The tensor representing a reward
    
        """
        if not self._is_final_state(data):
            reward = torch.tensor([0.0])
        elif self._is_win(data):
            reward = torch.tensor([1.0])
        elif self._is_draw(data):
            reward = torch.tensor([0.5])
        elif self._is_loss(data):
            reward = torch.tensor([0.0])
        else:
            raise Exception("no reward returned")
    
        return reward    

    def _get_feature_mapping(self):
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

    def _was_alive_previous_turn(self, data):
        game_id = data["game"]["id"]
        my_id = data["you"]["id"]

        return data["turn"] - 1 == self.turns[(game_id, my_id)]

    def _is_final_state(self, data):
        snakes = data["board"]["snakes"]

        return len(snakes) <= 1

    def _is_win(self, data):
        my_id = data["you"]["id"]
        snakes = data["board"]["snakes"]

        return len(snakes) == 1 and snakes[0]["id"] == my_id

    def _is_draw(self, data):
        snakes = data["board"]["snakes"]

        return not snakes and self._was_alive_previous_turn(data)

    def _is_loss(self, data):
        my_id = data["you"]["id"]
        snakes = data["board"]["snakes"]

        return (len(snakes) == 1 and snakes[0]["id"] != my_id) or (
            not snakes and not self._was_alive_previous_turn(data)
        )
