.PHONY: build test clean dev lint fmt check release

VERSION := $(shell cat VERSION)

build:
	cargo build --workspace

release:
	cargo build --release --workspace

test:
	cargo test --workspace

clean:
	cargo clean

dev:
	cargo watch -x 'run --package synapse-api'

lint:
	cargo clippy --workspace -- -D warnings

fmt:
	cargo fmt --all

check:
	cargo fmt --all -- --check
	cargo clippy --workspace -- -D warnings
	cargo test --workspace

docker-build:
	docker build -t synapse:$(VERSION) -f docker/Dockerfile .
	docker build -t synapse-trainer:$(VERSION) -f docker/Dockerfile.trainer .

docker-dev:
	docker compose -f docker/docker-compose.yml up --build

.DEFAULT_GOAL := build
