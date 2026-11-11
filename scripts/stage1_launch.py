"""Start amendment 56's Stage 1 on Modal's side (deploy cmca/midtrain/cpt_modal.py as cmca-stage1 first): the treated
arm at once, the control once the treated arm has logged 10 healthy steps. Returns immediately; nothing here needs
the launching machine afterwards.

  MODAL_PROFILE=engine APP_NAME=cmca-stage1 RETRIES=1 MAX_HOURS=12 modal deploy cmca/midtrain/cpt_modal.py
  MODAL_PROFILE=engine python scripts/stage1_launch.py mistral24 "<extra trainer flags, e.g. Cho's optimizer>"
"""

import os
import sys

import modal

family, extra = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
arm = modal.Function.from_name(os.environ.get("APP_NAME", "cmca-stage1"), "arm")  # cmca-stage1-nano: amendment 59
t = arm.spawn(family, "treated", "treated", 1e-6, 5e-6, extra)
c = arm.spawn(family, "control", "control", 1e-6, 5e-6, extra, wait_for=f"{family}/treated_mt")
print("treated", t.object_id, "control", c.object_id)
