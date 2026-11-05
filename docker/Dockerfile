# ifran — the AGNOS training control plane, built from source.
#
#   docker build -f docker/Dockerfile --build-arg VERSION="$(cat VERSION)" -t ifran .
#   docker run --rm -v ifran-data:/var/lib/ifran ifran runs
#
# Stage 1 installs the cyrius toolchain at the cyrius.cyml pin and builds the
# static ifran binary. Stage 2 is the AGNOS base with that binary added.
# ifran is a CLI, not a server: each `docker run` is one command against the
# state in /var/lib/ifran (ifran.db, run logs, keys under .ifran/). Jobs run
# whatever `bin` names, so mount or layer the sibling binaries a job needs
# (see Dockerfile.trainer).

FROM debian:trixie-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# The toolchain pin comes from cyrius.cyml, never from this file.
COPY cyrius.cyml VERSION ./
RUN CYRIUS_VERSION="$(grep '^cyrius = ' cyrius.cyml | head -1 | sed 's/cyrius = "\(.*\)"/\1/')" \
    && echo "Installing Cyrius $CYRIUS_VERSION" \
    && curl -sSf https://raw.githubusercontent.com/MacCracken/cyrius/main/scripts/install.sh \
        | CYRIUS_VERSION="$CYRIUS_VERSION" sh
ENV PATH="/root/.cyrius/bin:${PATH}"

COPY . .
# --locked: lib/ and cyrius.lock must be exactly what the pinned tags resolve.
RUN cyrius deps --locked \
    && mkdir -p build \
    && cyrius build src/main.cyr build/ifran \
    && ./build/ifran --version

# Runtime stage — AGNOS base (Debian trixie + tini)
FROM ghcr.io/maccracken/agnosticos:latest

ARG VERSION=dev
LABEL org.opencontainers.image.title="ifran"
LABEL org.opencontainers.image.description="The AGNOS training control plane: jobs, signed checkpoint store, datasets, sweeps, evals, preference data"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.source="https://github.com/MacCracken/ifran"
LABEL org.opencontainers.image.licenses="GPL-3.0-only"
LABEL org.opencontainers.image.base.name="ghcr.io/maccracken/agnosticos:latest"

USER root
RUN groupadd -g 1002 ifran \
    && useradd -u 1002 -g ifran -d /var/lib/ifran -M -s /usr/sbin/nologin ifran \
    && mkdir -p /var/lib/ifran \
    && chown ifran:ifran /var/lib/ifran \
    && chmod 0750 /var/lib/ifran

COPY --from=builder /src/build/ifran /usr/local/bin/ifran

USER ifran
# keys default to $HOME/.ifran; ifran.db and relative logdirs land in the cwd
ENV HOME=/var/lib/ifran
WORKDIR /var/lib/ifran
VOLUME ["/var/lib/ifran"]

ENTRYPOINT ["tini", "--", "/usr/local/bin/ifran"]
CMD ["--help"]
