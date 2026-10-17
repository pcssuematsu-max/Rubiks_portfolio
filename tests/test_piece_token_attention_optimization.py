import math
import unittest

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI, torch


@unittest.skipIf(torch is None, 'PyTorch is unavailable')
class PieceTokenAttentionOptimizationTests(unittest.TestCase):
    @staticmethod
    def _reference_attention(model, x, params):
        indices = model.layers['Aff1'].piece_feature_indices
        tokens = torch.stack([
            params['W1'].index_select(1, torch.as_tensor(piece, dtype=torch.long))
            @ x.index_select(0, torch.as_tensor(piece, dtype=torch.long))
            for piece in indices
        ])
        q = torch.einsum('od,pdb->pob', params['WQ1'], tokens)
        k = torch.einsum('od,pdb->pob', params['WK1'], tokens)
        scores = torch.einsum('pdb,qdb->bpq', q, k) / math.sqrt(q.shape[1])
        attention = torch.softmax(scores, dim=-1)
        attended = torch.einsum('bpq,qdb->pdb', attention, tokens)
        return params['WV1'] @ attended.sum(dim=0) + params['B1'].unsqueeze(1)

    @staticmethod
    def _inputs(model, batch_size):
        rng = np.random.default_rng(77)
        x = torch.tensor(
            rng.normal(size=(model.ips, batch_size)).astype('f'),
            requires_grad=True,
        )
        params = {
            key: torch.tensor(model.params[key].copy(), requires_grad=True)
            for key in ('W1', 'B1', 'WQ1', 'WK1', 'WV1')
        }
        return x, params

    def test_forward_and_gradients_match_reference_for_3x3_and_7x7(self):
        for cube_size, batch_size, expected_tokens in (
            (3, 1, 26), (3, 3, 26), (7, 1, 218), (7, 2, 218),
        ):
            with self.subTest(cube_size=cube_size, batch_size=batch_size):
                model = Rubiks_3_AI(
                    [16, 16], cube_size=cube_size,
                    use_transformer_attention=True,
                    transformer_attention_dim=8,
                    transformer_attention_token_mode='piece',
                )
                self.assertEqual(len(model.layers['Aff1'].piece_feature_indices), expected_tokens)
                reference_x, reference_params = self._inputs(model, batch_size)
                optimized_x, optimized_params = self._inputs(model, batch_size)

                reference = self._reference_attention(model, reference_x, reference_params)
                optimized = model._torch_piece_token_attention(optimized_x, optimized_params)
                torch.testing.assert_close(optimized, reference, rtol=2e-5, atol=2e-5)

                reference.square().mean().backward()
                optimized.square().mean().backward()
                torch.testing.assert_close(
                    optimized_x.grad, reference_x.grad, rtol=2e-4, atol=2e-5,
                )
                for key in reference_params:
                    torch.testing.assert_close(
                        optimized_params[key].grad, reference_params[key].grad,
                        rtol=2e-4, atol=2e-5,
                    )


if __name__ == '__main__':
    unittest.main()
