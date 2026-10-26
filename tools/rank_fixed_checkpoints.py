"""Rank saved checkpoints only across identical fixed fixture sets.

The report never loads or replaces parameters.  It keeps difficult bands
visible so a gain on easy cases cannot hide a regression on compositions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


PRIORITY_BANDS = ('low-effect/very-long-sequence', 'high-effect/short-sequence')


def rank_checkpoints(records):
    groups = defaultdict(list)
    for record in records:
        if record.get('recordType') != 'ai_result':
            continue
        key = (record['fixtureSetId'], record['aiIndex'])
        groups[key].append(record)
    output = []
    for (fixture_set_id, ai_index), rows in sorted(groups.items()):
        fixture_orders = {tuple(case['fixtureId'] for case in row['cases']) for row in rows}
        if len(fixture_orders) != 1:
            raise ValueError('A fixture set ID contains different fixture orders')
        scored = []
        for row in rows:
            cases = row['cases']
            by_band = {}
            for band in PRIORITY_BANDS:
                matching = [case for case in cases if case['band'] == band]
                by_band[band] = {
                    'solved': sum(bool(case['directSolved']) for case in matching),
                    'trials': len(matching),
                }
            total = sum(bool(case['directSolved']) for case in cases)
            score = tuple(by_band[band]['solved'] for band in PRIORITY_BANDS) + (total,)
            scored.append({
                'runId': row.get('runId'),
                'checkpoint': row.get('checkpoint'),
                'score': list(score),
                'directSolved': total,
                'trials': len(cases),
                'byPriorityBand': by_band,
            })
        scored.sort(key=lambda item: tuple(item['score']), reverse=True)
        output.append({
            'fixtureSetId': fixture_set_id,
            'aiIndex': ai_index,
            'fixtureIds': list(next(iter(fixture_orders))),
            'rankedCheckpoints': scored,
        })
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    records = []
    for path in args.inputs:
        with path.open(encoding='utf-8') as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    report = {'priorityBands': list(PRIORITY_BANDS), 'groups': rank_checkpoints(records)}
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + '\n', encoding='utf-8')
    else:
        print(encoded)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
