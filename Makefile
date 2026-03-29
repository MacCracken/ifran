.PHONY: build release test test-unit test-integration test-coverage \
       clean dev lint fmt format-check check \
       security-scan docs version-sync \
       ci-build ci-test ci-docs \
       docker-build docker-dev docker-release

VERSION := $(shell cat VERSION)

# === Build ===

build:
	cargo build --workspace

release:
	cargo build --release --workspace

clean:
	cargo clean

# === Development ===

dev:
	cargo watch -x 'run --package ifran-api'

# === Quality ===

fmt:
	cargo fmt --all

format-check:
	cargo fmt --all -- --check

lint:
	cargo clippy --workspace -- -D warnings

check: format-check lint test

# === Testing ===

test:
	cargo test --workspace

test-unit:
	cargo test --workspace --lib

test-integration:
	cargo test --workspace --test '*'

test-coverage:
	cargo tarpaulin --all-features --out xml --output-dir coverage/ \
		--fail-under 70 --skip-clean

# === Security ===

security-scan:
	cargo audit

# === Documentation ===

docs:
	cargo doc --workspace --no-deps

# === CI Targets ===

ci-build: check build
ci-test: test lint format-check security-scan
ci-docs: docs

# === Docker ===

docker-build:
	docker build -t ifran:$(VERSION) -f docker/Dockerfile .
	docker build -t ifran-trainer:$(VERSION) -f docker/Dockerfile.trainer .

docker-dev:
	docker compose -f docker/docker-compose.yml up --build

docker-release:
	docker build -t ghcr.io/maccracken/ifran:$(VERSION) -f docker/Dockerfile .
	docker tag ghcr.io/maccracken/ifran:$(VERSION) ghcr.io/maccracken/ifran:latest

# === Version ===

version-set:
	@./scripts/version-set.sh $(v)

version-sync:
	@./scripts/version-set.sh $(VERSION)

.DEFAULT_GOAL := build
