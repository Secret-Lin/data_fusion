"""Bounded cached clips: singleton failures plus fixed motion controls, no inference."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from tools.run_real_ots_ablation import run


async def main():
    # Diagnostics cover known failure windows and independent control windows.
    # No per-session fusion parameters; these are development clips, not test claims.
    for session,start,duration in [('stable-0',21.,2.),('stable-0',29.,3.),
                                   ('stable-0',55.,2.),('Test-0',12.,2.),('Test-4',37.5,2.),
                                   ('stable-1',10.,3.),('Test-1',20.,3.),
                                   ('Test-2',10.,3.),('Test-2',30.,3.),('Test-3',20.,3.)]:
        args = SimpleNamespace(run_dir=Path('artifacts/new_real_full')/session/'results_0',
                               start=start,duration=duration,full_sequence=False,leave_one_out=False,
                               singleton_guard=True,calibration_dir=Path('artifacts/new_real_reference/results_1'),
                               output_parent=Path('artifacts/singleton_guard_clips'))
        cached = [json.loads(line) for line in (args.run_dir/'observations.jsonl').read_text().splitlines()]
        if not any(start <= r['frame_id']/30 < start+duration for r in cached):
            print(f'{session}/{start}: no frontend observations; retained as a missing-input interval, not an accuracy result', flush=True)
            continue
        if await run(args):
            raise RuntimeError(f'Failed variant in {session}/{start}')


if __name__ == '__main__':
    asyncio.run(main())
