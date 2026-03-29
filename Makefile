.PHONY: build release test test-unit test-integration test-coverage \
       clean dev lint fmt format-check check \
       security-scan docs bench bench-history version-sync \
       ci-build ci-test ci-docs \
       docker-build docker-dev docker-release

VERSION := $(shell cat VERSION)

# === Build ===

build:
	cargo build --all-features

release:
	cargo build --release --all-features

clean:
	cargo clean

# === Development ===

dev:
	cargo run --bin ifran-server --all-features

# === Quality ===

fmt:
	cargo fmt

format-check:
	cargo fmt -- --check

lint:
	cargo clippy --all-features --all-targets -- -D warnings

check: format-check lint test

# === Testing ===

test:
	cargo test --all-features

test-unit:
	cargo test --all-features --lib

test-integration:
	cargo test --all-features --test '*'

test-coverage:
	cargo tarpaulin --all-features --out xml --output-dir coverage/ \
		--fail-under 70 --skip-clean

# === Benchmarks ===

bench:
	cargo bench --all-features

bench-history:
	./scripts/bench-history.sh

# === Security ===

security-scan:
	cargo audit
	cargo deny check

# === Documentation ===

docs:
	RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps

# === CI Targets ===

ci-build: check build
ci-test: test lint format-check security-scan
ci-docs: docs

# === Docker ===

docker-build:
	docker build -t ifran:$(VERSION) -f docker/Dockerfile .

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
