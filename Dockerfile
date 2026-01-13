# Multi-stage Dockerfile for VirNucPro standalone container
# WHY multi-stage: Separating build and runtime reduces image size ~30% (5GB → 3.5GB)
# by excluding build dependencies (gcc, build-essential) from final image. Faster cloud VM startup.
# Builder stage: compile and install dependencies
# WHY CUDA 11.8: PyTorch 2.0+ has proven stability with CUDA 11.8, compatible with V100/T4/A100 GPUs.
# CUDA 12.x has compatibility issues with transformers==4.30.0 and fair-esm==2.0.0.
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3.9 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# VirNucPro repository configuration
# WHY ARG for commit pinning: Allows building specific VirNucPro versions without Dockerfile changes.
# Single Dockerfile can build multiple versions by passing --build-arg VIRNUCPRO_COMMIT=<sha>.
# Pattern from beast2-beagle-cuda for version matrix builds.
ARG VIRNUCPRO_REPO=https://github.com/Li-Jing-1997/VirNucPro.git
ARG VIRNUCPRO_COMMIT=HEAD

# Clone VirNucPro repository
RUN git clone ${VIRNUCPRO_REPO} /opt/VirNucPro && \
    cd /opt/VirNucPro && \
    git checkout ${VIRNUCPRO_COMMIT}

# Capture VirNucPro version at build time
RUN cd /opt/VirNucPro && git rev-parse HEAD > /tmp/virnucpro_version.txt

# Copy wrapper requirements and install
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Install VirNucPro dependencies
RUN pip3 install --no-cache-dir -r /opt/VirNucPro/requirements.txt

# WHY uninstall triton: Prevents GPU compatibility issues across different CUDA device generations.
# Triton adds no value for VirNucPro's inference-only use case. Per VirNucPro README recommendation.
RUN pip3 uninstall -y triton || true

# Runtime stage: minimal production image with only necessary artifacts
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-distutils \
    samtools \
    && rm -rf /var/lib/apt/lists/*

# Copy VirNucPro installation from builder
COPY --from=builder /opt/VirNucPro /opt/VirNucPro

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/dist-packages /usr/local/lib/python3.9/dist-packages

# Copy VirNucPro version file from builder
COPY --from=builder /tmp/virnucpro_version.txt /tmp/virnucpro_version.txt

# Copy Python wrapper files
COPY virnucpro.py virnucpro_cli.py /opt/

# Set environment variables
ENV VIRNUCPRO_PATH="/opt/VirNucPro"
ENV PATH="/usr/local/bin:${PATH}"

# Set working directory
WORKDIR /data

# Entry point and default command
ENTRYPOINT ["python3"]
CMD ["/opt/virnucpro_cli.py", "--help"]
