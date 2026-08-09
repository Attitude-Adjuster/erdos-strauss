# Erdos-Straus certificate census -- portable runner.
#
#   docker build -t erdos .
#   docker run --rm -v "$PWD/out:/out" erdos 0 1000000000000        # 10^12 smoke
#   docker run --rm -v "$PWD/out:/out" erdos 0 1000000000000000     # 10^15
#
# The scanner is one translation unit with no dependencies beyond libstdc++ and
# OpenMP, so this image is deliberately thin. It is NOT a prebuilt binary: the
# entrypoint compiles on the machine it will run on and refuses to proceed if the
# self-test fails. That is not caution theatre --
#
#   * the Makefile defaults to -march=native, and a binary built on one microarch
#     and shipped to another dies with SIGILL. Compiling at start keeps the native
#     tuning (worth ~10% here) without making the image host-specific.
#   * RUNBOOK requires `make check` per machine, per compiler, PER FLAG SET, because
#     what it guards against is miscompilation and unsafe optimization, not logic
#     bugs. An image that skipped it would be the exact hazard it warns about.
#
# Compilation costs ~8s against runs measured in hours. Set NATIVE=0 to build a
# portable binary instead (heterogeneous fleets, or if the host CPU is unknown).

FROM debian:12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        g++ make python3 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /erdos
COPY rung_scan.cpp Makefile run8.sh merge.py verify_cert.py analyze_census.py ./
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x run8.sh /usr/local/bin/docker-entrypoint.sh

# Census output lands here; mount a volume over it or the results die with the
# container. At 10^15 with EMIT_RESIDUAL=0 this is a few hundred MB.
VOLUME /out
ENV OUTDIR=/out/census

ENTRYPOINT ["docker-entrypoint.sh"]
