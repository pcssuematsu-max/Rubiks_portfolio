"""Compare direct solve rates by physical Effect type without changing training.

Normal solve logs retain effectClass from the next recorded solve onward.  The
older compact logs omit setup moves, so their missing classes cannot be
reconstructed.  Fixed benchmark rows do retain scrambles and can be classified
offline.  Results from one fixed setup are descriptive, not a population rate.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.experiment_log import setup_effect_metrics


def _category(search_mode, value_loss_type):
    if search_mode == 'search3':
        return 'Search3'
    return {
        'myloss': 'Original Search2',
        'steps_to_goal': 'steps_to_goal',
        'myloss2_pairwise': 'pairwise',
    }.get(value_loss_type, search_mode or 'unknown')


def _length_band(length):
    if length < 20:
        return '0-19'
    if length < 30:
        return '20-29'
    if length < 40:
        return '30-39'
    return '40+'


def _rows(path, benchmark_cube_size=4):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('recordType') == 'ai_result':
                settings = record.get('settings') or {}
                for case in record.get('cases', ()):  # same fixed state for every AI
                    yield {
                        'aiIndex': record['aiIndex'],
                        'model': settings.get('model'),
                        'category': _category(settings.get('searchMode'), settings.get('valueLossType')),
                        'effect': case.get('setupEffect'),
                        'scramble': case.get('scramble'),
                        'cubeSize': benchmark_cube_size,
                        'length': len(case.get('scramble') or ()),
                        'direct': bool(case.get('directSolved')),
                        'fallback': False,
                        'fixtureId': case.get('fixtureId'),
                    }
                continue
            if 'searchSucceeded' not in record:
                continue
            settings = record.get('aiSettings') or {}
            search2 = settings.get('search2') or {}
            yield {
                'aiIndex': record.get('aiIndex'),
                'model': settings.get('model'),
                'category': _category(record.get('searchMode'), search2.get('valueLossType')),
                'effect': record.get('setupEffect'),
                'scramble': record.get('setup'),
                'cubeSize': record.get('cubeSize'),
                'length': int(record.get('setupMoveCount') or 0),
                'direct': bool(record.get('searchSucceeded')),
                'fallback': bool(record.get('fallbackUsed') and record.get('succeeded')),
                'fixtureId': None,
            }


def summarize(rows, min_trials=20):
    groups = defaultdict(list)
    overall_groups = defaultdict(list)
    missing = 0
    count = 0
    for row in rows:
        count += 1
        effect = row.get('effect') or {}
        effect_class = effect.get('effectClass')
        if not effect_class and row.get('scramble'):
            # Only fixed benchmark rows retain a route.  Cached effect analysis
            # is done here, outside the solve and training loops.
            size = int(row['cubeSize'])
            measured = setup_effect_metrics(f'cube-{size}x{size}', size, row['scramble'])
            effect_class = (measured or {}).get('effectClass')
        if not effect_class:
            missing += 1
            continue
        key = (
            row['category'], row['model'], effect_class,
            _length_band(row['length']),
        )
        groups[key].append(row)
        overall_groups[(effect_class, _length_band(row['length']))].append(row)
    result = []
    for (category, model, effect_class, length_band), items in groups.items():
        if len(items) < min_trials:
            continue
        direct = sum(item['direct'] for item in items)
        result.append({
            'category': category,
            'model': model,
            'effectClass': effect_class,
            'lengthBand': length_band,
            'trials': len(items),
            'aiCount': len({item['aiIndex'] for item in items}),
            'fixtureCount': len({item['fixtureId'] for item in items if item['fixtureId']}),
            'directSuccessCount': direct,
            'directSuccessRate': round(direct / len(items), 4),
            'fallbackSuccessCount': sum(item['fallback'] for item in items),
        })
    result.sort(key=lambda item: (item['category'], item['model'] or '', item['lengthBand'], -item['trials'], item['effectClass']))
    overall_result = []
    for (effect_class, length_band), items in overall_groups.items():
        if len(items) < min_trials:
            continue
        direct = sum(item['direct'] for item in items)
        overall_result.append({
            'effectClass': effect_class,
            'lengthBand': length_band,
            'trials': len(items),
            'aiCount': len({item['aiIndex'] for item in items}),
            'fixtureCount': len({item['fixtureId'] for item in items if item['fixtureId']}),
            'directSuccessCount': direct,
            'directSuccessRate': round(direct / len(items), 4),
            'fallbackSuccessCount': sum(item['fallback'] for item in items),
        })
    overall_result.sort(key=lambda item: (item['lengthBand'], -item['trials'], item['effectClass']))
    return {
        'rowCount': count,
        'missingEffectClassCount': missing,
        'minTrials': min_trials,
        'overallGroups': overall_result,
        'groups': result,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Experiment JSONL or paired benchmark JSONL')
    parser.add_argument('--benchmark-cube-size', type=int, choices=(3, 4), default=4)
    parser.add_argument('--min-trials', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.min_trials < 1:
        parser.error('--min-trials must be positive')
    report = summarize(_rows(args.input, args.benchmark_cube_size), args.min_trials)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
    else:
        print(encoded)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
