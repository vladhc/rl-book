from random import randint
from asciimatics.screen import Screen
import rl
import time


def evaluate(env, graph):

    def _gui(screen):

        root = graph.root()
        state = env.reset()
        stack = [(root, 0, state)]

        done = False

        while not done:
            action, stack = rl.get_action(
                    graph,
                    stack,
                    state,
                    greedy=True)
            next_state, reward, done, debug = env.step(action)
            state = next_state

            screen.print_at(env.render(mode='ansi'), 0, 0)
            time.sleep(1)

            # screen.print_at('Hello world!',
                            # randint(0, screen.width),
                            # randint(0, screen.height),
                            # colour=randint(0, screen.colours - 1),
                            # bg=randint(0, screen.colours - 1))
            ev = screen.get_key()
            if ev in (ord('Q'), ord('q')):
                return
            screen.refresh()

    Screen.wrapper(_gui)
