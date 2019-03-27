class Task:

    def __init__(
            self,
            name,
            primitive_action=None,
            params=None,
            state_tr=lambda state: state,
            term_predicate=lambda state: True):
        self.name = name

        # For the composite actions
        self._actions = []
        self._params = params
        self._state_tr = state_tr
        self._term_predicate = term_predicate

        # For the primitive actions
        self._primitive_action = primitive_action

    def __iadd__(self, action):
        self._actions.append(action)
        return self

    def get(self, name, params=None):
        for action in self._actions:
            if action.name == name and action.get_params() == params:
                return action

    def get_params(self):
        return self._params

    def transform_state(self, state):
        return self._state_tr(state)

    def is_primitive(self):
        return len(self._actions) == 0

    def get_primitive(self):
        return self._primitive_action

    def get_actions(self):
        return self._actions

    def is_done(self, state):
        if self.is_primitive():
            return True
        state = self.transform_state(state)
        return self._term_predicate(state)
