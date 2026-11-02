"""Champion/challenger operations. Separate from training on purpose: no training
run can make itself champion (ADR 0005 D8).
python scripts/champion.py --compare
python scripts/champion.py --promote --actor p.raman
python scripts/champion.py --rollback --actor p.raman
"""
import argparse
import sys
from aegis_ml.training.registry import compare, promote_challenger, rollback

ap = argparse.ArgumentParser()
ap.add_argument("--compare", action="store_true")
ap.add_argument("--promote", action="store_true")
ap.add_argument("--rollback", action="store_true")
ap.add_argument("--actor", default="unknown")
a = ap.parse_args()

if a.rollback:
    d = rollback(a.actor)
elif a.promote:
    d = promote_challenger(a.actor)
else:
    d = compare()

print(d.reason)
sys.exit(0 if d.changed else 1)
