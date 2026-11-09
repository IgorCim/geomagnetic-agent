"""One recording run; original suites and chat boundaries remain unchanged."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def build_plan():
    base = runpy.run_path(str(ROOT / 'tools/run_chat_acceptance.py'))['build_plan']()
    focused = runpy.run_path(str(ROOT / 'tools/run_focused_review.py'))['build_plan']()
    # The eight-message formula chain is already part of focused (43 messages).
    # Keep original failures visible: never replace a historical case with a pass.
    return [dict(phase, id=f'base_{phase["id"]}') for phase in base] + [
        dict(phase, id=f'focused_{phase["id"]}') for phase in focused]


def run(brain):
    helper = runpy.run_path(str(ROOT / 'tools/run_chat_acceptance.py'))
    plan = build_plan()
    print('Combined recording:', len(plan), 'independent chats;',
          sum(len(p['cases']) for p in plan), 'cards including setup/blocked cases.')
    return helper['run_acceptance'](brain, plan=plan)
