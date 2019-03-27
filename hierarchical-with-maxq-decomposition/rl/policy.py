import numpy as np


def get_action(graph, stack, state, epsilon=0.0):
    while True:
        assert len(stack) > 0
        task, params, *_ = stack[-1]
        if task.is_done(state):
            stack.pop()
        else:
            break
    while True:
        task, *_ = stack[-1]
        if task.is_primitive():
            break
        # Evaluate next action
        next_task = _policy(graph, task, state, epsilon)
        stack.append((next_task, 0, state))
    action, *_ = stack[-1]  # Primitive action

    # increase tick of every action on stack
    for i in range(len(stack)):
        task, ticks, state = stack[i]
        ticks += 1
        stack[i] = (task, ticks, state)

    return action.get_primitive(), stack


def print_stack(stack):
    for i, record in enumerate(stack):
        action, ticks, _ = record
        print("{}. {}({}): {}".format(i, action.name, action.get_params(), ticks))


def _policy(graph, task, state, epsilon):
    best_action = graph.get_best_action(task, state)
    if np.random.uniform() > epsilon:
        return best_action

    actions = filter(
            lambda a: a.is_primitive() or not a.is_done(state),
            task.get_actions())
    actions = list(actions)
    actions.remove(best_action)

    if len(actions) == 0:
        return best_action

    rand_idx = np.random.choice(len(actions))
    return actions[rand_idx]
