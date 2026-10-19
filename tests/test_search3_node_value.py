import unittest
from types import SimpleNamespace

import numpy as np

from cube.search3_engine import Node, Search3Engine, sigmoid


class Search3NodeValueTests(unittest.TestCase):
    def test_unvisited_actions_use_normalized_value_after_refresh(self):
        node = Node(np.array([0.5, 0.5], dtype='f'), value=0.0)
        node.visited[0] = 1
        node.S = 1

        node.val[0] = 1.0
        node.set_value_from_logit(3.0)
        self.assertAlmostEqual(node.value, sigmoid(3.0))
        self.assertEqual(node.select_node(C=0.0), 0)

        node.val[0] = 0.0
        node.set_value_from_logit(-3.0)
        self.assertAlmostEqual(node.value, sigmoid(-3.0))
        self.assertEqual(node.select_node(C=0.0), 1)

    def test_duplicate_leaf_reuses_normalized_node_value(self):
        ai = SimpleNamespace(
            search3_C=0.5,
            _predict_search2=lambda features: np.tile(
                np.array([[0.0], [0.0], [-3.0]], dtype='f'),
                (1, features.shape[1]),
            ),
        )
        engine = Search3Engine.__new__(Search3Engine)
        engine.ai = ai
        engine.ips = 1
        engine.node_cache = {}
        engine.prediction_cache = {}
        parent = Node(np.array([0.5, 0.5], dtype='f'), value=0.0)
        leaves = [
            {
                'state_key': 'same-state',
                'feature': np.array([0.0], dtype='f'),
                'parent_node': parent,
                'move_label': move,
                'path_nodes': [parent],
            }
            for move in ('A', 'B')
        ]

        engine._predict_leaf_batch(leaves)

        child = engine.node_cache['same-state']
        self.assertIs(parent.children['A'], child)
        self.assertIs(parent.children['B'], child)
        self.assertAlmostEqual(child.value, sigmoid(-3.0))
        self.assertTrue(all(leaf['node_value'] == child.value for leaf in leaves))

    def test_cached_current_state_refreshes_normalized_value(self):
        engine = Search3Engine.__new__(Search3Engine)
        engine.ai = SimpleNamespace(search3_C=0.5)
        engine._state_key = lambda: 'same-state'
        engine._predict_current_state = lambda: np.array(
            [[0.0], [0.0], [2.0]], dtype='f',
        )
        cached = Node(np.array([0.5, 0.5], dtype='f'), value=-2.0)
        engine.node_cache = {'same-state': cached}

        node, raw_value = engine._create_node_from_current_state()

        self.assertIs(node, cached)
        self.assertEqual(raw_value, 2.0)
        self.assertAlmostEqual(node.value, sigmoid(2.0))


if __name__ == '__main__':
    unittest.main()
