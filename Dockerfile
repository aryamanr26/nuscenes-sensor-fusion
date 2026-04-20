# nuScenes colorization - GPU-ready development/runtime image with host UID/GID mapping
#
# Build (from project root):
#   docker build -t nuscenes-colorization \
#     --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --build-arg USER_NAME=$(whoami) .
#
# Run:
#   docker run --gpus all -it --rm \
#     -v /home/aryamanr/nuscenes-colorization:/workspace/nuscenes-colorization \
#     -v /mnt/workspace/datasets/nuscenes:/data/nuscenes \
#     nuscenes-colorization

ARG CUDA_VERSION=12.1.1
ARG CUDNN_VERSION=8
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu22.04

ARG USER_NAME=nuscenes
ARG USER_ID=1000
ARG GROUP_ID=1000

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# --------------------
# System deps (root)
# --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    ca-certificates \
    curl \
    wget \
    git \
    unzip \
    build-essential \
    software-properties-common \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure `python` resolves to Python 3 for script compatibility.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /workspace/nuscenes-colorization

COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && "${VIRTUAL_ENV}/bin/pip" install --upgrade pip setuptools wheel \
    && "${VIRTUAL_ENV}/bin/pip" install -r /tmp/requirements.txt

COPY . /workspace/nuscenes-colorization

# --------------------
# Non-root user + passwordless sudo
# --------------------
RUN getent group "${GROUP_ID}" >/dev/null 2>&1 || groupadd -g "${GROUP_ID}" "${USER_NAME}" \
    && useradd -m -l -u "${USER_ID}" -g "${GROUP_ID}" -s /bin/bash "${USER_NAME}" \
    && usermod -aG video "${USER_NAME}" \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/"${USER_NAME}" \
    && chmod 0440 /etc/sudoers.d/"${USER_NAME}"

# Make sure mounted workspace is accessible by the mapped host user.
RUN chown -R "${USER_ID}:${GROUP_ID}" /workspace

USER ${USER_NAME}
WORKDIR /workspace/nuscenes-colorization
ENV PATH="/home/${USER_NAME}/.local/bin:/opt/venv/bin:${PATH}"

CMD ["/bin/bash"]
