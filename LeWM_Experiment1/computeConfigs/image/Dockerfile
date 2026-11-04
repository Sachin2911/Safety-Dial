# syntax=docker/dockerfile:1
FROM vastai/pytorch:2.7.1-cuda-12.6.3-py311-22.04@sha256:3e1a0ce60011d85d6dcafa96ca543286fcfcb52509941e3471dfd30ac9619815 AS platform

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

LABEL org.opencontainers.image.title="LeWM and Anchored Planning runtime" \
      org.opencontainers.image.source="https://github.com/Sachin2911/Safety-Dial" \
      org.opencontainers.image.version="0.1.0" \
      ai.lewm.ap-revision="984327424ca7a2e52c75a8279197750d9d0b7c8e"

ENV PATH="/venv/main/bin:${PATH}" \
    AP_ROOT=/opt/ap-reference/code/ap \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    GALLIUM_DRIVER=llvmpipe \
    LP_NUM_THREADS=1 \
    SDL_VIDEODRIVER=dummy \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates git libegl1 libgl1 libglib2.0-0 libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN git init /opt/ap-reference \
    && git -C /opt/ap-reference remote add origin \
       https://github.com/daybraeklaxry/Aim-Short-to-Reach-Far.git \
    && git -C /opt/ap-reference sparse-checkout init --cone \
    && git -C /opt/ap-reference sparse-checkout set code/ap \
    && git -C /opt/ap-reference fetch --depth=1 --filter=blob:none origin \
       984327424ca7a2e52c75a8279197750d9d0b7c8e \
    && git -C /opt/ap-reference checkout --detach FETCH_HEAD \
    && test "$(git -C /opt/ap-reference rev-parse HEAD)" = \
       984327424ca7a2e52c75a8279197750d9d0b7c8e

COPY requirements.ap.txt constraints.cuda.txt verify_runtime.py /opt/lewm-runtime/
# The base's pip metadata points at an unavailable conda build directory.
# Reinstall the same version from PyPI so the full lock is portable.
RUN /venv/main/bin/python -m pip install --no-cache-dir --no-deps --force-reinstall pip==25.3
RUN dpkg-query -W > /opt/lewm-runtime/os-packages.txt

# Resolve once on the rented builder and export requirements.lock before release.
FROM platform AS resolve
RUN /venv/main/bin/python -m pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cu126 \
       -c /opt/lewm-runtime/constraints.cuda.txt -r /opt/lewm-runtime/requirements.ap.txt \
    && /venv/main/bin/python -m pip install --no-deps --no-build-isolation \
       -e /opt/ap-reference/code/ap \
    && /venv/main/bin/python -m pip check \
    && /venv/main/bin/python /opt/lewm-runtime/verify_runtime.py \
    && /venv/main/bin/python -m pip freeze --all --exclude anchored-planning \
       > /opt/lewm-runtime/requirements.lock

# An absent lock is deliberately a build error for the publishable target.
FROM platform AS runtime
COPY requirements.lock /opt/lewm-runtime/requirements.lock
RUN /venv/main/bin/python -m pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cu126 \
       -c /opt/lewm-runtime/requirements.ap.txt -c /opt/lewm-runtime/constraints.cuda.txt \
       -r /opt/lewm-runtime/requirements.lock \
    && /venv/main/bin/python -m pip install --no-deps --no-build-isolation \
       -e /opt/ap-reference/code/ap \
    && /venv/main/bin/python -m pip check \
    && /venv/main/bin/python /opt/lewm-runtime/verify_runtime.py \
    && /venv/main/bin/python -m pip freeze --all --exclude anchored-planning \
       > /tmp/requirements.installed \
    && diff -u /opt/lewm-runtime/requirements.lock /tmp/requirements.installed \
    && rm /tmp/requirements.installed

WORKDIR /workspace
# Inherit Vast's entrypoint. The existing SSH template invokes that bootstrap.
