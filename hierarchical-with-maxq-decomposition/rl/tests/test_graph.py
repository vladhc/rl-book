from unittest import TestCase
import rl

class TestGraph(TestCase):

    def testValueCompositeAction(self):
        bottom = rl.Task("↓", 0)
        right = rl.Task("→", 0)
        root = rl.Task("root", term_predicate=lambda x: False)
        root += bottom
        root += right
        graph = rl.Graph(root)

        graph.set_v(bottom, (0, 0), -1)
        graph.set_v(right,  (0, 0), -100)
        graph.set_c(root, (0, 0), bottom, 0.3)

        # V = Q(root, state, best_action(s))
        # best_action((0, 0)) = bottom
        # V = V(bottom, (0, 0)) + C(root, (0,0), bottom)
        # V = -1 + 0.3 = -0.7
        self.assertEqual(graph.get_v(root, (0, 0)), -0.7)
