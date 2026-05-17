FROM rust:1-slim-trixie AS builder
WORKDIR /usr/src/hex-table
COPY . .
RUN cargo install --path . -F nn,burn --bin train-daemon

FROM debian:trixie-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libvulkan1 \
        vulkan-tools \
        libx11-6 \
        libxext6 \
        libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Courtesy of Claude:
#   wgpu needs a Vulkan ICD JSON pointing at the NVIDIA Vulkan driver. The
#   NVIDIA Container Toolkit normally injects this when the `graphics` driver
#   capability is enabled, but Cloud Run's GPU mount only documents CUDA and
#   does not create this file. The ICD library itself (libGLX_nvidia.so.0)
#   lives in /usr/local/nvidia/lib64, which Cloud Run prepends to
#   LD_LIBRARY_PATH, so a relative `library_path` resolves correctly.
RUN mkdir -p /usr/share/vulkan/icd.d
COPY etc/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json

COPY --from=builder /usr/local/cargo/bin/train-daemon /usr/local/bin/train-daemon
CMD ["train-daemon"]
