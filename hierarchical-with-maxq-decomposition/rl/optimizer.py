import numpy as np
from .graph import get_default_graph

epsilon = 0.00000001


def _move(x, target, alpha, episode, max_episodes):
    global epsilon
    k = min(1.0, float(episode) / float(max_episodes))
    alpha = max(epsilon, alpha * (1.0 - k))
    return (1.0 - alpha) * x + alpha * target


class Optimizer:

    def __init__(
            self,
            graph=None,
            learning_rate=0.02,
            discount_factor=0.9,
            steps=1000):
        if graph is None:
            graph = get_default_graph()
        self._graph = graph
        assert self._graph is not None
        self._learning_rate = learning_rate
        self._discount_factor = discount_factor
        self._episodes = steps

    def optimize(self, episode_num, stack, next_state, done):
        self._episode = episode_num

        # update actions on stack in reverse order
        for idx in reversed(range(len(stack))):
            action, _, state = stack[idx]
            if not action.is_done(next_state) and not done:
                return
            if action.is_primitive():
                parent_action, _, _ = stack[idx - 1]
                reward = 0 if parent_action.is_done(next_state) else -1
                self._update_v(action, state, reward)
            else:
                sub_action, ticks, state = stack[idx + 1]
                self._update_c(
                        action,
                        state,
                        sub_action,
                        ticks,
                        next_state)

    def _update_v(self, action, state, reward):
        assert action.is_primitive()
        v = self._graph.get_v(action, state)
        v = _move(v, reward, self._learning_rate, self._episode, self._episodes)
        self._graph.set_v(action, state, v)

    def _update_c(self, action, state, sub_action, ticks, next_state):
        assert not action.is_primitive()
        target = np.power(self._discount_factor, ticks) * self._graph.get_v(action, next_state)
        c = self._graph.get_c(action, state, sub_action)
        c = _move(c, target, self._learning_rate, self._episode, self._episodes)
        self._graph.set_c(action, state, sub_action, c)
