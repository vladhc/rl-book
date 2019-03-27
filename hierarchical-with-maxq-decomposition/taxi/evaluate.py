import sys
import time
from six import StringIO

import rl


def _print_env(env, fn, pass_idx=3, dest_idx=0, mode='human'):
    outfile = StringIO() if mode == 'ansi' else sys.stdout

    outfile.write("+----------+\n")
    for y in range(0, 5):
        outfile.write("|")
        for x in range(0, 5):
            state = env.unwrapped.encode(y, x, pass_idx, dest_idx)
            outfile.write(fn(state) + " ")
        outfile.write("|")
        outfile.write("\n")
    outfile.write("+----------+\n")

    if mode != 'human':
        s = outfile.getvalue()
        outfile.close()
        return s


def print_best_actions(env, graph, mode='human'):
    def fn(state):
        action = graph.root()
        while not action.is_primitive():
            action = graph.get_best_action(action, state)
        return action.name
    return _print_env(env, fn, mode=mode)


def print_v(env, graph, action, pass_idx=3, dest_idx=0):
    _print_env(env, lambda state: "{: 1.4f}".format(
        graph.get_v(action, state)))


def print_c(env, graph, action, child_action, pass_idx=3, dest_idx=0):
    _print_env(env, lambda state: "{: 1.4f}".format(
        graph.get_c(action, state, child_action)))


def print_q(env, graph, action, child_action, pass_idx=3, dest_idx=0):
    _print_env(env, lambda state: "{: 1.4f}".format(
        graph.get_q(action, state, child_action)))


def evaluate(env, graph):
    episode = 0
    while True:
        print("== Episode {} ==".format(episode))
        _play_episode(env, graph)
        episode += 1


def _play_episode(env, graph):
    root = graph.root()
    state = env.reset()
    stack = [(root, 0, state)]

    done = False
    step = 0

    while not done:
        if root.is_done(state):
            break

        action, stack = rl.get_action(
                graph,
                stack,
                state,
                epsilon=0.0)
        state, reward, done, debug = env.step(action)

        env_str = env.render(mode='ansi').getvalue()
        env_str = env_str.split('\n')
        action_str = print_best_actions(env, graph, mode='ansi')
        action_str = action_str.split('\n')

        print("Step {}".format(step))
        print("  state: {}".format(list(env.unwrapped.decode(state))))
        idx = 0
        for env_line, action_line in zip(env_str, action_str):
            stack_str = ""
            if idx < len(stack):
                task, ticks, _ = stack[idx]
                stack_str = "({} {})".format(task.name, ticks)
                idx += 1
            print("  {}  {}  {}".format(env_line, action_line, stack_str))
        print()

        step += 1

        time.sleep(1)
