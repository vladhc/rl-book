import gym


class Task:

    def __init__(self, name,
            primitiveAction=None,
            paramValues=[],
            termPredicate=lambda state, params : True):
        self._name = name
        self._actions = []
        self._primitiveAction = primitiveAction
        self._paramValues = paramValues
        self._termPredicate = termPredicate

    def __iadd__(self, action):
        self._actions.append(action)
        return self

    def is_primitive(self):
        return len(self._actions) == 0

    def get_primitive(self):
        return self._primitiveAction

    def policy(self, state, params):
        assert len(self._actions) > 0, "action {} has no children".format(self._name)
        return self._actions[0], None

    def is_done(self, state, params):
        return self._termPredicate(state, params)


def get_action(stack, state):
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
        next_task, next_task_params = task.policy(state, params)
        stack.append((next_task, next_task_params))
    action, _ = stack.pop()
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
        taxi_y, taxi_x, pass_idx, dest_idx = env.decode(state)
        x, y = params
        return x == taxi_x and y == taxi_y
    gridCoords = [(x, y) for x in range(5) for y in range(5)]
    navigate = Task("navigate",
            paramValues=gridCoords,
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
