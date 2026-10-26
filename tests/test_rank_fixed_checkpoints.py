import unittest

from tools.rank_fixed_checkpoints import rank_checkpoints


class RankFixedCheckpointsTests(unittest.TestCase):
    def test_difficult_band_breaks_easy_case_tie(self):
        def row(run, difficult, easy):
            return {
                'recordType': 'ai_result', 'fixtureSetId': 'same',
                'aiIndex': 2, 'runId': run, 'checkpoint': {'sha256': run},
                'cases': [
                    {'fixtureId': 'long', 'band': 'low-effect/very-long-sequence',
                     'directSolved': difficult},
                    {'fixtureId': 'easy', 'band': 'low-effect/short-sequence',
                     'directSolved': easy},
                ],
            }
        result = rank_checkpoints([row('old', False, True), row('new', True, False)])
        self.assertEqual(result[0]['rankedCheckpoints'][0]['runId'], 'new')
        self.assertEqual(result[0]['rankedCheckpoints'][0]['score'], [1, 0, 1])

    def test_different_fixture_sets_are_never_compared(self):
        a = {'recordType': 'ai_result', 'fixtureSetId': 'set-a', 'aiIndex': 2,
             'cases': [{'fixtureId': 'a', 'band': 'low-effect/very-long-sequence', 'directSolved': True}]}
        b = {'recordType': 'ai_result', 'fixtureSetId': 'set-b', 'aiIndex': 2,
             'cases': [{'fixtureId': 'b', 'band': 'low-effect/very-long-sequence', 'directSolved': False}]}
        self.assertEqual(len(rank_checkpoints([a, b])), 2)


if __name__ == '__main__':
    unittest.main()
