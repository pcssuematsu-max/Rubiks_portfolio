import unittest

import numpy as np

from ai.losses import Myloss
from ai.rubiks_ai import Rubiks_3_AI, torch


class OriginalSearch2MarginTests(unittest.TestCase):
    def test_endpoint_margin_applies_to_every_preceding_state(self):
        values = np.array([[1.0, 2.0, 3.0]], dtype='f')
        loss = Myloss(margin=1.0)

        actual = loss.forward(values, [0, 3])
        logits = np.array([2.0, 3.0, 3.0])
        expected_probabilities = np.exp(logits - np.max(logits))
        expected_probabilities /= np.sum(expected_probabilities)

        self.assertAlmostEqual(actual, -np.log(expected_probabilities[-1]), places=6)
        np.testing.assert_allclose(
            loss.backward(),
            (expected_probabilities - np.array([0.0, 0.0, 1.0])).reshape(1, -1),
            rtol=1e-6,
        )
        self.assertGreater(actual, Myloss(margin=0.0).forward(values, [0, 3]))

    @unittest.skipIf(torch is None, 'PyTorch unavailable')
    def test_torch_and_numpy_endpoint_margin_match(self):
        values = np.array([[1.0, 2.0, 3.0, 0.0, 4.0]], dtype='f')
        indices = [0, 3, 5]
        expected = Myloss(margin=1.0)
        numpy_loss = expected.forward(values, indices)
        tensor = torch.tensor(values.reshape(-1), requires_grad=True)
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)

        torch_loss = ai._torch_search2_value_loss_softmax(tensor, indices, margin=1.0)
        torch_loss.backward()

        self.assertAlmostEqual(float(torch_loss.detach()), numpy_loss, places=6)
        np.testing.assert_allclose(
            tensor.grad.detach().numpy(), expected.backward().reshape(-1), rtol=1e-6,
        )

    @unittest.skipIf(torch is None, 'PyTorch unavailable')
    def test_search3_rank_loss_uses_the_value_weight_of_each_route(self):
        values = np.array([[1.0, 2.0, 3.0, 0.0, 4.0]], dtype='f')
        indices = [0, 3, 5]
        weights = np.array([[0.2, 0.2, 0.2, 0.8, 0.8]], dtype='f')
        numpy_rank = Myloss()
        numpy_loss = numpy_rank.forward(values, indices, weights)
        tensor = torch.tensor(values.reshape(-1), requires_grad=True)
        ai = Rubiks_3_AI.__new__(Rubiks_3_AI)

        torch_loss = ai._torch_search2_value_loss_softmax(
            tensor, indices, state_weights=torch.tensor(weights.reshape(-1)),
        )
        torch_loss.backward()

        self.assertAlmostEqual(float(torch_loss.detach()), numpy_loss, places=6)
        np.testing.assert_allclose(
            tensor.grad.detach().numpy(), numpy_rank.backward().reshape(-1), rtol=1e-6,
        )
