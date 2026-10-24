"""Baseline for the existing AegisFlow M1 schema.

This revision records the pre-existing schema. It intentionally does not
create or drop any M1 tables.
"""

revision = "m1_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
