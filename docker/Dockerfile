FROM rust:1.89-bookworm AS builder

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
RUN cargo build --release --package ifran-api --package ifran-cli

# Runtime stage — AGNOS base OS (AI/ML operating system layer)
FROM ghcr.io/maccracken/agnosticos:latest

ARG VERSION
LABEL org.opencontainers.image.title="Ifran"
LABEL org.opencontainers.image.description="LLM controller for pulling, managing, and training language models"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.source="https://github.com/MacCracken/ifran"
LABEL org.opencontainers.image.licenses="AGPL-3.0-only"
LABEL org.opencontainers.image.base.name="ghcr.io/maccracken/agnosticos:latest"

# AGNOS base includes ca-certificates and tini
USER root
RUN groupadd -g 1002 ifran && \
    useradd -u 1002 -g ifran -m -s /bin/bash ifran

# Create data directories
RUN mkdir -p /var/lib/ifran/models /var/lib/ifran/checkpoints /var/lib/ifran/cache && \
    chown -R ifran:ifran /var/lib/ifran

COPY --from=builder /build/target/release/ifran-api /usr/local/bin/ifran-server
COPY --from=builder /build/target/release/ifran /usr/local/bin/ifran

USER ifran
EXPOSE 8420 8421

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/ifran", "status"] || exit 1

ENTRYPOINT ["tini", "--"]
CMD ["ifran-server"]
