#!/usr/bin/env bash
# Execute on the rented VM with Docker and the NVIDIA Container Toolkit available.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

image="${LEWM_IMAGE:-ghcr.io/sachin2911/lewm-ap:0.1.0}"
docker info >/dev/null
container_id=""
lock_temp=""
cleanup() {
    if [[ -n "$container_id" ]]; then docker rm "$container_id" >/dev/null; fi
    if [[ -n "$lock_temp" ]]; then rm -f -- "$lock_temp"; fi
}
trap cleanup EXIT

if [[ ! -e requirements.lock ]]; then
    docker build --platform linux/amd64 --target resolve --tag "${image}-resolve" .
    container_id="$(docker create --entrypoint /bin/true "${image}-resolve")"
    lock_temp="$(mktemp ./requirements.lock.XXXXXX)"
    docker cp "$container_id:/opt/lewm-runtime/requirements.lock" "$lock_temp"
    # A release lock must remain usable in a fresh build, without local wheel paths.
    if grep -Eq '(^-e | @ file:| @ git\+)' "$lock_temp"; then
        echo 'Resolve local/editable package references before releasing this lock.' >&2
        exit 1
    fi
    test -s "$lock_temp"
    mv -- "$lock_temp" requirements.lock
    lock_temp=""
    docker rm "$container_id" >/dev/null
    container_id=""
fi

docker build --platform linux/amd64 --target runtime --tag "$image" .
printf 'Built %s. Run publish_image.py on this GPU host to validate and publish.\n' "$image"
