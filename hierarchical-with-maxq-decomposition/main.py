import gym
import numpy as np
from collections import defaultdict

epsilon = 0.01


class Task:

    def __init__(self, _id,
            primitiveAction=None,
            paramValues=[None],
            stateTr=lambda state : state,
            termPredicate=lambda state, params : True):
        self._id = _id

        # For the composite actions
        self._actions = []
        self._paramValues = paramValues
        self._stateTr = stateTr
        self._termPredicate = termPredicate
        self._c = {}  # Completition function: param → state → C
        for params in self._paramValues:
            self._c[params] = defaultdict(float)  # state → value
        self._bindedActions = set()  # Actions with specific param values

        # For the primitive actions
        self._v = defaultdict(float)  # state → value
        self._primitiveAction = primitiveAction

    def __iadd__(self, action):
        self._actions.append(action)
        for param in action.get_params():
            self._bindedActions.add((action, param))
        return self

    def get_params(self):
        return self._paramValues

    def is_primitive(self):
        return len(self._actions) == 0

    def get_primitive(self):
        return self._primitiveAction

    def value(self, state, params):
        if self.is_primitive():
            return self._v[state]
        stateTr = self._stateTr(state)
        c = self._c[params][stateTr]  # params → state → C
        bestAction, bestActionParams = self._best_binded_action(state, params)
        return bestAction.value(state, bestActionParams) + c


    def policy(self, state, params, greedy):
        global epsilon
        assert len(self._actions) > 0, "action {} has no children".format(self._name)

        bestBindedAction = self._best_binded_action(state, params)

        if greedy or np.random.uniform() > epsilon:
            return bestBindedAction

        bindedActions = self._bindedActions.copy()
        bindedActions.remove(bestBindedAction)
        bindedActions = list(bindedActions)
        assert len(bindedActions) > 0, "Action {} has only one child".format(self._id)
        randIdx = np.random.choice(len(bindedActions))
        return bindedActions[randIdx]

    def _best_binded_action(self, state, params):
        bestBindedAction = None  # tuple action+params
        bestBindedActionValue = -float('inf')

        for action, actionParams in self._bindedActions:
            bindedActionValue = action.value(state, actionParams)
            if bindedActionValue > bestBindedActionValue:
                bestBindedAction = (action, actionParams)
                bestBindedActionValue = bindedActionValue

        assert bestBindedAction is not None
        return bestBindedAction

    def is_done(self, state, params):
        state = self._stateTr(state)
        return self._termPredicate(state, params)


def get_action(stack, state, greedy=False):
    while True:
        assert len(stack) > 0
        task, params = stack[-1]
        if task.is_done(state, params):
            stack.pop()
        else:
            break
    while True:
        task, params = stack[-1]
        if task.is_primitive():
            break
        next_task, next_task_params = task.policy(state, params, greedy)
        stack.append((next_task, next_task_params))
    action, _ = stack.pop()  # Primitive action
    return action.get_primitive(), stack


# Task specific graph
def create_graph(env, done_fn):
    south = Task("south", 0)
    north = Task("north", 1)
    east = Task("east", 2)
    west = Task("west", 3)
    pickup = Task("pickup", 4)
    dropoff = Task("dropoff", 5)

    def navigateTerminated(state, params):
        taxi_x, taxi_y = state
        x, y = params
        return x == taxi_x and y == taxi_y

    def navigateState(state):
        taxi_y, taxi_x, pass_idx, dest_idx = env.decode(state)
        return taxi_x, taxi_y
    gridCoords = [(x, y) for x in range(5) for y in range(5)]
    navigate = Task("navigate",
            paramValues=gridCoords,
            stateTr=navigateState,
            termPredicate=navigateTerminated)
    navigate += north
    navigate += south
    navigate += east
    navigate += west

    get = Task("get")
    get += pickup
    get += navigate

    put = Task("put")
    put += dropoff
    put += navigate

    root = Task("root", termPredicate=done_fn)
    root += put
    root += get

    return root


env = gym.make('Taxi-v2')
done = False
root = create_graph(env.unwrapped, lambda state, params : done)
stack = [(root, None)]

state = env.reset()
for _ in range(10):
    env.render()
    action, stack = get_action(stack, state)
    state, reward, done, debug = env.step(action)
    env.render()

env.close()
