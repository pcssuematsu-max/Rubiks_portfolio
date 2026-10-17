import unittest

import numpy as np

from ai.rubiks_ai import Rubiks_3_AI, torch


class TransformerLinearSkipTests(unittest.TestCase):
    def _model(self, *, enabled=True, residual=True, token_mode='piece', skip_variance=0.125):
        return Rubiks_3_AI(
            [16, 16],
            residual=residual,
            use_transformer_attention=True,
            transformer_attention_dim=8,
            transformer_attention_token_mode=token_mode,
            transformer_linear_skip=enabled,
            transformer_linear_skip_variance=skip_variance,
        )

    def test_zero_initialized_skip_preserves_existing_predictions(self):
        baseline = self._model(enabled=False)
        model = self._model(skip_variance=0.0)
        for key in baseline.params:
            model.params[key][:] = baseline.params[key]
        x = np.random.default_rng(24).normal(size=(model.ips, 2)).astype('f')

        np.testing.assert_allclose(
            model.predict(x, policy=True, value=True, retain_cache=True),
            baseline.predict(x, policy=True, value=True, retain_cache=True),
            rtol=1e-5, atol=1e-5,
        )
        self.assertEqual(model.params['WSkip1'].shape, (16, model.ips))
        self.assertEqual(model.params['BSkip1'].shape, (16,))
        self.assertFalse(np.any(model.params['WSkip1']))

    def test_linear_base_variance_uses_fan_in(self):
        model = self._model()
        weights = model.params['WSkip1']
        observed = float(np.var(weights) * weights.shape[1])
        self.assertAlmostEqual(observed, 0.125, delta=0.125 * 0.15)
        self.assertFalse(np.any(model.params['BSkip1']))

    def test_numpy_skip_branch_receives_gradients(self):
        for residual, token_mode in ((True, 'piece'), (False, 'piece'), (True, 'hidden')):
            with self.subTest(residual=residual, token_mode=token_mode):
                model = self._model(residual=residual, token_mode=token_mode)
                x = np.random.default_rng(44).normal(size=(model.ips, 2)).astype('f')
                model.predict(x, policy=True, value=True, retain_cache=True)
                model._backprop_through_trunk_layers(np.ones((16, 2), dtype='f'))
                self.assertGreater(float(np.linalg.norm(model.first_layer_linear_skip.dW)), 0.0)
                self.assertGreater(float(np.linalg.norm(model.first_layer_linear_skip.dB)), 0.0)

    @unittest.skipIf(torch is None, 'PyTorch is unavailable')
    def test_torch_and_numpy_forward_agree_with_nonzero_skip(self):
        model = self._model()
        model.params['WSkip1'][:] = np.random.default_rng(13).normal(
            scale=0.01, size=model.params['WSkip1'].shape,
        ).astype('f')
        x = np.random.default_rng(15).normal(size=(model.ips, 2)).astype('f')
        numpy_out = model.predict(x, policy=True, value=True, retain_cache=True)
        with torch.no_grad():
            torch_out = model._torch_forward_policy_value(
                torch.as_tensor(x), model._torch_params_from_numpy(torch.device('cpu')),
            ).numpy()
        np.testing.assert_allclose(numpy_out, torch_out, rtol=1e-4, atol=1e-4)

    @unittest.skipIf(torch is None, 'PyTorch is unavailable')
    def test_torch_training_updates_skip_weight(self):
        model = self._model()
        params = model._torch_trainable_params(torch.device('cpu'))
        x = torch.as_tensor(np.random.default_rng(37).normal(size=(model.ips, 2)).astype('f'))
        model._torch_forward_policy_value(x, params).sum().backward()
        self.assertGreater(float(torch.linalg.vector_norm(params['WSkip1'].grad)), 0.0)
        before = model.params['WSkip1'].copy()
        model._apply_torch_grad_updates(model._torch_grad_dict(params))
        self.assertFalse(np.array_equal(model.params['WSkip1'], before))


if __name__ == '__main__':
    unittest.main()
