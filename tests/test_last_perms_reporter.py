import unittest
from types import SimpleNamespace

from core.myperm_keys import MypermKey
from managers.last_perms_reporter import LastPermsReporter


class _ReporterCube:
    def __init__(self):
        self.myperms = {
            MypermKey('Known', 0): ('R', 'U', 'F'),
        }


class _EffectAdapter:
    def analyze_effect(self, _cube, moves):
        return SimpleNamespace(moved_count = len(moves))


class LastPermsReporterTests(unittest.TestCase):
    def setUp(self):
        self.frame = SimpleNamespace(
            cube = _ReporterCube(),
            last_perms = {
                'Known': {('R', 'U'), ('R', 'U', 'F', 'L')},
                'LP:unregistered': {('L',)},
            },
            last_perms_changed_number = {'Known': 1, 'LP:unregistered': 1},
            puzzle_adapter = _EffectAdapter(),
        )
        self.reporter = LastPermsReporter(self.frame)

    def test_key_infos_include_registered_and_found_lengths(self):
        infos = {info['key']: info for info in self.reporter.lp_key_infos()}

        self.assertEqual(infos['Known']['myperm_length'], 3)
        self.assertEqual(infos['Known']['found_minimum_length'], 2)
        self.assertEqual(infos['Known']['found_lengths'], (2, 4))
        self.assertEqual(infos['Known']['effect_count'], 3)
        self.assertIsNone(infos['LP:unregistered']['myperm_length'])
        self.assertEqual(infos['LP:unregistered']['found_minimum_length'], 1)

    def test_key_infos_sort_myperms_before_other_keys(self):
        infos = self.reporter.lp_key_infos()

        self.assertEqual([info['key'] for info in infos], ['Known', 'LP:unregistered'])


if __name__ == '__main__':
    unittest.main()
