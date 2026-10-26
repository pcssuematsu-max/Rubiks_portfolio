"""Summarize recent calibrated-Value drift, label conflicts, and gradients.

Use this after each training run to compare the last two equal windows.  The
fixed validation is the deciding signal; training MAE alone changes with the
sampled replay distribution.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]


def _mean(records, *keys):
    values = []
    for record in records:
        value = record
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return statistics.mean(values) if values else None


def _window(records):
    if not records:
        return None
    metrics = {
        'from': records[0]['timestamp'],
        'to': records[-1]['timestamp'],
        'count': len(records),
        'trainingMae': _mean(records, 'stepsToGoalRegression', 'maePerState'),
        'trainingDataCount': _mean(records, 'trainingDataCount'),
        'valueGradientShare': _mean(records, 'gradientMetrics', 'valueGradientShareMean'),
        'gradientRms': _mean(records, 'gradientMetrics', 'globalRmsMean'),
        'conflictingStateCount': _mean(
            records, 'stepsToGoalRegression', 'valueSelection', 'conflictingIntermediateStateCount',
        ),
        'excludedAmbiguousValueStateCount': _mean(
            records, 'stepsToGoalRegression', 'valueSelection', 'excludedAmbiguousValueStateCount',
        ),
        'candidateValueStateCount': _mean(records, 'stepsToGoalRegression', 'stateCount'),
        'fixedMaeByLength': {
            length: _mean(records, 'fixedValidation', 'byLength', length, 'valueMae')
            for length in ('12', '24', '36', '48', '60')
        },
    }
    count = metrics['candidateValueStateCount']
    conflicts = metrics['conflictingStateCount']
    metrics['conflictingStatesPerValueState'] = (
        conflicts / count if count and conflicts is not None else None
    )
    return metrics


def analyze(records, ai_indices=(20, 22), window_size=20):
    grouped = defaultdict(list)
    for record in records:
        if record.get('aiIndex') in ai_indices and record.get('search2ValueLossType') == 'steps_to_goal':
            grouped[record['aiIndex']].append(record)
    result = {'windowSize': window_size, 'ai': {}}
    for index in ai_indices:
        rows = grouped[index]
        previous = _window(rows[-2 * window_size:-window_size])
        recent = _window(rows[-window_size:])
        warnings = []
        if previous and recent:
            for length in ('24', '36', '48', '60'):
                old = previous['fixedMaeByLength'][length]
                new = recent['fixedMaeByLength'][length]
                if old is not None and new is not None and new > old:
                    warnings.append(f'fixed_{length}_mae_increased')
            old = previous['conflictingStatesPerValueState']
            new = recent['conflictingStatesPerValueState']
            if old is not None and new is not None and new > old:
                warnings.append('conflicting_label_fraction_increased')
        result['ai'][str(index)] = {
            'previous': previous, 'recent': recent, 'warnings': warnings,
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=ROOT / 'exports' / 'ai-learning-history.json')
    parser.add_argument('--ai-indices', default='20,22')
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.window_size < 1:
        parser.error('Window size must be positive')
    try:
        indices = tuple(int(part.strip()) for part in args.ai_indices.split(','))
    except ValueError:
        parser.error('AI indices must be comma-separated integers')
    rows = json.loads(args.history.read_text(encoding='utf-8'))['records']
    report = analyze(rows, indices, args.window_size)
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
    else:
        print(encoded)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
