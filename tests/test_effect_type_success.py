import unittest

from tools.analyze_effect_type_success import summarize


class EffectTypeSuccessTests(unittest.TestCase):
    def test_separates_direct_success_and_route_length(self):
        def row(length, direct, fallback):
            return {
                'aiIndex': 1, 'model': 'linear', 'category': 'Search3',
                'effect': {'effectClass': 'C2[2]+W2-2s'},
                'scramble': None, 'length': length,
                'direct': direct, 'fallback': fallback, 'fixtureId': None,
            }

        report = summarize([
            row(18, True, False), row(18, False, True), row(37, False, True),
        ], min_trials=1)
        self.assertEqual(report['missingEffectClassCount'], 0)
        by_length = {item['lengthBand']: item for item in report['overallGroups']}
        self.assertEqual(by_length['0-19']['directSuccessRate'], 0.5)
        self.assertEqual(by_length['0-19']['fallbackSuccessCount'], 1)
        self.assertEqual(by_length['30-39']['directSuccessRate'], 0.0)


if __name__ == '__main__':
    unittest.main()
