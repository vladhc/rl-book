from unittest import TestCase
import rl


class TestOptimizer(TestCase):

    def testPrimitiveAction(self):
        a = rl.Task("↓", 0)
        root = rl.Task("root", term_predicate=lambda x: False)
        root += a
        graph = rl.Graph(root)

        optimizer = rl.Optimizer(graph, learning_rate=0.5)
        state = (0, 0)
        next_state = (0, 1)

        for v in [-0.5, -0.75, -0.875]:
            stack = [
                    (root, 1, state),
                    (a,    1, state),
                    ]
            optimizer.optimize(stack, next_state, False)
            self.assertEqual(graph.get_v(a, state), v)

    def testCompositeAction(self):
        down = rl.Task("↓", 0)
        root = rl.Task("root")
        root += down
        graph = rl.Graph(root)

        optimizer = rl.Optimizer(
                graph,
                learning_rate=0.75,
                discount_factor=1.0)
        state0 = (0, 0)
        state = (0, 3)
        next_state = (0, 4)

        graph.set_v(down, state, -1.0)
        graph.set_v(down, next_state, -0.25)
        graph.set_c(root, state, down, 1.0)
        graph.set_c(root, next_state, down, 2.0)

        stack = [
                (root, 4, state0),
                (down, 1, state),
                ]

        optimizer.optimize(stack, next_state, False)

        # Parent task is finished => V of the primitive action:
        # target = 0.0 => 0.25 * -1.0 + 0.75 * 0.0 = -0.25
        self.assertEqual(graph.get_v(down, state), -0.25)

        # target C(root, state, down) = V(root, next_state) =
        #   = Q(root, next_state, down) =
        #   = V(down, next_state) + C(root, next_state, down) =
        #   = -0.25 + 2.0 = 1.75
        # C(root, state, down) = 0.25 * 1.0 + 0.75 * 1.75 = 1.5625
        self.assertEqual(graph.get_c(root, state, down), 1.5625)
