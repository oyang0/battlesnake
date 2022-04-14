import numpy as np

"""
Battlesnake efficiently updatable neural network.
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
        accumulator = self._ft(features) if features is not None else self.accumulator
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
