.PHONY: up down logs ps db-init seed verify test lint clean reset

COMPOSE = docker compose -p aegisflow

up:
	$(COMPOSE) up -d --build
	@echo "stack up ΓÇö run 'make ps' to check health"

down:
	$(COMPOSE) down

reset:
	$(COMPOSE) down -v

ps:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f --tail=100

db-init:
	docker compose exec -T postgres psql -U $${DB_USER} -d $${DB_NAME} -f
	/docker-entrypoint-initdb.d/01_init.sql

seed:
	python scripts/seed_db.py

data:
	python scripts/download_datasets.py

verify:
	python scripts/verify_m1.py

test:
	pytest tests/ -v

lint:
	black . && flake8 .

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} +
# ---- M2 targets: append below the M1 block ----
.RECIPEPREFIX = >
PY ?= python
M2_PATH := PYTHONPATH=services/ingestion/src:ml/src:.
M2_COMPOSE := docker compose -f docker-compose.yml -f docker-compose.m2.yml

up-m2:
> $(M2_COMPOSE) up -d --build ingestion graph-job

migrate:
> $(M2_PATH) alembic upgrade head

seed-fraud:
> $(M2_PATH) $(PY) db/seed/seed_known_fraud.py

ingest:
> $(M2_PATH) uvicorn ingestion.main:app --reload --port 8001

graph-once:
> $(M2_PATH) $(PY) -m aegis_ml.features.graph_features --once

features-pit:
> dvc repro features_pit

parity:
> $(M2_PATH) PARITY_REDIS_URL=redis://localhost:6379/15 pytest -q tests/m2/unit/test_m2_parity.py

test-m2:
> $(M2_PATH) pytest -q tests/m2/unit tests/m2/api

test-m2-int:
> $(M2_PATH) pytest -q tests/m2/integration

verify-m2:
> $(M2_PATH) $(PY) scripts/verify_m2.py

m2: test-m2 test-m2-int verify-m2

.PHONY: up-m2 migrate seed-fraud ingest graph-once features-pit parity test-m2 test-m2-int verify-m2 m2

# ---- M3 targets: append below the M2 block. .RECIPEPREFIX applies only to rules
# written after it, so M2's tab-indented rules keep working and every M3 rule
# survives being copied out of this PDF. ----
.RECIPEPREFIX = >
PY ?= python
M3_PATH := PYTHONPATH=ml/src:.

prepare-ieee:
> dvc repro prepare_ieee

train-fraud:
> dvc repro train_fraud

compare:
> $(M3_PATH) $(PY) scripts/champion.py --compare

promote:
> $(M3_PATH) $(PY) scripts/champion.py --promote --actor $(USER)

rollback:
> $(M3_PATH) $(PY) scripts/champion.py --rollback --actor $(USER)

typologies:
> $(M3_PATH) $(PY) -m aegis_ml.evaluation.typology_eval

test-m3:
> $(M3_PATH) $(PY) -m pytest -q tests/m3/unit

test-m3-int:
> $(M3_PATH) $(PY) -m pytest -q tests/m3/integration

verify-m3:
> $(M3_PATH) $(PY) scripts/verify_m3.py

m3: test-m3 train-fraud typologies test-m3-int verify-m3

.PHONY: prepare-ieee train-fraud compare promote rollback typologies \
 test-m3 test-m3-int verify-m3 m3

