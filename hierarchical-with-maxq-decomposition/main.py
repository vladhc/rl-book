
import numpy as np
import gym

import rl
from taxi import evaluate, print_best_actions, print_v, print_c, print_q


n_episodes = 10000
max_steps = 200


# Taxi-v2 specific graph
def create_graph(env):
    env = env.unwrapped
    # Creating graph from bottom to the top
    target_coords = [(0, 0), (4, 0), (0, 4), (3, 4)]

    # Navigate tasks (parameterized)
    navigate_tasks = [
            create_navigate_task(env, target_coord)
            for target_coord in target_coords
    ]

    # Dropoff
    def dropoff_state(state):
        taxi_y, taxi_x, pass_idx, dest_idx = env.decode(state)
        return taxi_x, taxi_y, dest_idx
    dropoff = rl.Task("D", 5, state_tr=dropoff_state)

    # Pickup
    def pickup_state(state):
        taxi_y, taxi_x, pass_idx, dest_idx = env.decode(state)
        return taxi_y, taxi_x, pass_idx
    pickup = rl.Task("P", 4)

    # Get task
    def pickup_state(state):
        taxi_y, taxi_x, pass_idx, dest_idx = env.decode(state)
        return taxi_x, taxi_y, pass_idx

    def picked_up(state):
        taxi_x, taxi_y, pass_idx = state
        return pass_idx == 4

    get = rl.Task("get", state_tr=pickup_state, term_predicate=picked_up)
    get += pickup
    for navigate_task in navigate_tasks:
        get += navigate_task

    # Put task
    def put_state(state):
        taxi_y, taxi_x, pass_idx, dest_idx = env.decode(state)
        return taxi_x, taxi_y, pass_idx, dest_idx

    def taxi_empty(state):
        taxi_x, taxi_y, pass_idx, dest_idx = state
        return pass_idx != 4

    put = rl.Task("put", state_tr=put_state, term_predicate=taxi_empty)
    put += dropoff
    for navigate_task in navigate_tasks:
        put += navigate_task

    # Root task
    # TODO: term_predicate
    root = rl.Task("root", term_predicate=lambda state: False)
    root += get
    root += put
    return root


def create_navigate_task(env, target_coord):
    env = env.unwrapped
    def coord_and_target_state(state):
        taxi_y, taxi_x, pass_idx, dest_idx = env.decode(state)
        return taxi_x, taxi_y, target_coord[0], target_coord[1]
    # Creating graph from bottom to the top
    south = rl.Task("↓", 0, state_tr=coord_and_target_state)
    north = rl.Task("↑", 1, state_tr=coord_and_target_state)
    east = rl.Task("→", 2, state_tr=coord_and_target_state)
    west = rl.Task("←", 3, state_tr=coord_and_target_state)

    # Navigate task
    def coord_state(state):
        taxi_y, taxi_x, pass_idx, dest_idx = env.decode(state)
        return taxi_x, taxi_y

    def reached_target(state):
        taxi_x, taxi_y = state
        x, y = target_coord
        return x == taxi_x and y == taxi_y

    navigate = rl.Task(
            "navigate",
            params=target_coord,
            state_tr=coord_state,
            term_predicate=reached_target)
    navigate += north
    navigate += south
    navigate += east
    navigate += west

    return navigate


def train(env, graph=None):
    global n_episodes
    global max_steps

    if graph is None:
        graph = rl.get_default_graph()
    assert graph is not None

    optimizer = rl.Optimizer(
            graph=graph,
            learning_rate=0.5,
            discount_factor=1.0,
            steps=n_episodes)
    root = graph.root()

    for i in range(n_episodes):
        state = env.reset()
        stack = [(root, 0, state)]  # action, ticks, state0
        step = 0

        try:
            for _ in range(max_steps):
                if root.is_done(state):
                    break
                k = min(1.0, float(i) / float(n_episodes * 0.33))
                k = 1.0 - k
                epsilon = 0.05 * k
                action, stack = rl.get_action(
                        graph,
                        stack,
                        state,
                        epsilon=epsilon)
                next_state, reward, done, debug = env.step(action)
                optimizer.optimize(i, stack, next_state, done)
                state = next_state
                step += 1
                if done:
                    break
        except KeyError as error:
            print('episode {}, step {}'.format(i, step))
            print('state: {}', list(env.unwrapped.decode(state)))
            env.render()
            rl.print_stack(stack)
            raise error

        print("{} solved in {} steps".format(i, step))
    env.render()
    print('== best_action ==')
    print_best_actions(env.unwrapped, graph)
    print('== V ==')
    print_v(env.unwrapped, graph, root)
    for sub_task in root.get_actions():
        print('== Q({}) =='.format(sub_task.name))
        print_q(env.unwrapped, graph, root, sub_task)
        print('== C({}) =='.format(sub_task.name))
        print_c(env.unwrapped, graph, root, sub_task)
    print('C map size: {}'.format(len(graph._c)))
    # for k, v in graph._c.items():
        # print(k, v)
    print('V map size: {}'.format(len(graph._v)))
    # for k, v in graph._v.items():
        # print(k, v)


if __name__ == '__main__':
    env = gym.make('Taxi-v2')
    # target_coord = (3, 4)
    root = create_graph(env)
    root = root.get("get")

    graph = rl.Graph(root)
    train(env, graph=graph)
    evaluate(env, graph)
    env.close()
