# Multi-stage Dockerfile for VirNucPro standalone container
# WHY multi-stage: Separating build and runtime reduces image size ~30% (5GB → 3.5GB)
# by excluding build dependencies (gcc, build-essential) from final image. Faster cloud VM startup.
# Builder stage: compile and install dependencies
# WHY CUDA 12.6: VirNucPro v2.0 requires PyTorch >= 2.8.0, which dropped CUDA 11.8 support.
# CUDA 12.6 is the minimum supported version for PyTorch 2.8.0. Compatible with V100/T4/A100/H100 GPUs.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# VirNucPro repository configuration
# WHY ARG for commit pinning: Allows building specific VirNucPro versions without Dockerfile changes.
# Single Dockerfile can build multiple versions by passing --build-arg VIRNUCPRO_COMMIT=<sha>.
# Pattern from beast2-beagle-cuda for version matrix builds.
ARG VIRNUCPRO_REPO=https://github.com/broadinstitute/virnucpro-broad.git
ARG VIRNUCPRO_COMMIT=HEAD

# Clone VirNucPro repository (refactored version with multi-GPU support)
RUN git clone ${VIRNUCPRO_REPO} /opt/VirNucPro && \
    cd /opt/VirNucPro && \
    git checkout ${VIRNUCPRO_COMMIT}

# Copy wrapper requirements and install
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install VirNucPro dependencies (refactored version includes all dependencies)
# WHY pre-install: flash-attn's setup.py imports packaging and torch at egg_info time,
# before pip resolves build dependencies. Install them so metadata generation succeeds.
RUN pip3 install --no-cache-dir packaging torch && \
    pip3 install --no-cache-dir -r /opt/VirNucPro/requirements.txt

# Capture VirNucPro version at build time
# WHY after pip install: virnucpro.__init__ imports yaml (pyyaml) which must be installed first.
RUN cd /opt/VirNucPro && \
    echo "$(python3 -c 'import sys; sys.path.insert(0, "/opt/VirNucPro"); from virnucpro import __version__; print(__version__)')-$(git rev-parse --short HEAD)" > /tmp/virnucpro_version.txt

# WHY uninstall triton: Prevents GPU compatibility issues across different CUDA device generations.
# Triton adds no value for VirNucPro's inference-only use case. Per VirNucPro README recommendation.
RUN pip3 uninstall -y triton || true

# Clean up Python packages to reduce layer size
RUN find /usr/local/lib/python3.10/dist-packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10/dist-packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10/dist-packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local/lib/python3.10/dist-packages -type f -name "*.pyc" -delete 2>/dev/null || true && \
    find /usr/local/lib/python3.10/dist-packages -type f -name "*.pyo" -delete 2>/dev/null || true

# Runtime stage: minimal production image with only necessary artifacts
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-distutils \
    samtools \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Copy VirNucPro installation from builder
COPY --from=builder /opt/VirNucPro /opt/VirNucPro

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Copy VirNucPro version file from builder
COPY --from=builder /tmp/virnucpro_version.txt /tmp/virnucpro_version.txt

# Copy Python wrapper files
COPY virnucpro.py virnucpro_cli.py /opt/

# Set execute permissions on Python wrapper files
RUN chmod +x /opt/virnucpro.py /opt/virnucpro_cli.py

# Set environment variables
# VIRNUCPRO_PATH points to installation directory (contains models)
# PYTHONPATH includes VirNucPro to enable 'python -m virnucpro' invocation
ENV VIRNUCPRO_PATH="/opt/VirNucPro"
ENV PYTHONPATH="/opt/VirNucPro"
ENV PATH="/usr/local/bin:${PATH}"

# Set working directory
WORKDIR /data

# Default command (no entrypoint for Nextflow compatibility)
CMD ["/bin/bash"]
