FROM rust:1.85-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY proto/ proto/

# Build release binaries
RUN cargo build --release --package synapse-api --package synapse-cli

# Runtime stage — AGNOS base OS (AI/ML operating system layer)
FROM ghcr.io/maccracken/agnosticos:latest

ARG VERSION
LABEL org.opencontainers.image.title="Synapse"
LABEL org.opencontainers.image.description="LLM controller for pulling, managing, and training language models"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.source="https://github.com/MacCracken/synapse"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.base.name="ghcr.io/maccracken/agnosticos:latest"

# AGNOS base includes ca-certificates and tini
USER root
RUN groupadd -g 1002 synapse && \
    useradd -u 1002 -g synapse -m -s /bin/bash synapse

# Create data directories
RUN mkdir -p /var/lib/synapse/models /var/lib/synapse/checkpoints /var/lib/synapse/cache && \
    chown -R synapse:synapse /var/lib/synapse

COPY --from=builder /build/target/release/synapse-api /usr/local/bin/synapse-server
COPY --from=builder /build/target/release/synapse /usr/local/bin/synapse

USER synapse
EXPOSE 8420 8421

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/synapse", "status"] || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["synapse-server"]
