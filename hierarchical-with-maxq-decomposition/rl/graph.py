from .task import Task
from collections import defaultdict


class Graph:

    def __init__(self, root):
        self._v = defaultdict(float)  # (primitiveAction, state) → value
        self._c = defaultdict(float)  # (action, state, subAction) → completition
        self._root = root
        self._prev_default_graph = None

    def root(self):
        return self._root

    def __enter__(self):
        global _default_graph
        self._prev_default_graph = _default_graph
        _default_graph = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _default_graph
        _default_graph = self._prev_default_graph

    def get_v(self, action, state):
        if action.is_primitive():
            state_tr = action.transform_state(state)
            return self._v[(action.name, state_tr)]
        best_action = self.get_best_action(action, state)
        return self.get_q(action, state, best_action)

    def get_q(self, action, state, child_action):
        c = self.get_c(action, state, child_action)
        v = self.get_v(child_action, state)
        return v + c

    def set_v(self, action, state, v):
        assert action.is_primitive()
        state_tr = action.transform_state(state)
        self._v[(action.name, state_tr)] = v

    def get_c(self, parent_action, state, action):
        assert isinstance(parent_action, Task)
        assert isinstance(action, Task)
        assert parent_action.get_actions().index(action) != -1
        state_tr = action.transform_state(state)
        return self._c[(parent_action.name, state_tr, action.name)]

    def set_c(self, parent_action, state, action, c):
        assert isinstance(parent_action, Task)
        assert isinstance(action, Task)
        assert parent_action.get_actions().index(action) != -1
        state_tr = action.transform_state(state)
        self._c[(parent_action.name, state_tr, action.name)] = c

    def get_best_action(self, parent_action, state):
        assert not parent_action.is_primitive()
        best_action = None
        best_v = -float('inf')

        actions = filter(
                lambda a: a.is_primitive() or not a.is_done(state),
                parent_action.get_actions())

        for action in actions:
            v = self.get_q(parent_action, state, action)
            if v > best_v:
                best_action = action
                best_v = v

        assert best_action is not None, "No best action for task {}".format(parent_action.name)
        return best_action


_default_graph = None


def get_default_graph():
    global _default_graph
    return _default_graph
