#!/usr/bin/env bash
set -e

# ------------------------------------------------------------
# Location:
#   HUSKY_VOICE_MODULE/docker/ros_noetic_docker/pc/run.sh
# ------------------------------------------------------------

IMAGE_NAME="husky_voice_noetic_cuda126:pc"
CONTAINER_NAME="husky_voice_noetic_pc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# From:
#   HUSKY_VOICE_MODULE/docker/ros_noetic_docker/pc
#
# To:
#   HUSKY_VOICE_MODULE/src
HOST_SRC="$(realpath "${SCRIPT_DIR}/../../../src")"

CONTAINER_WS="/catkin_ws"
CONTAINER_SRC="${CONTAINER_WS}/src"

ROS_VARIANT="${ROS_VARIANT:-ros-base}"
FORCE_DOCKER_BUILD="${FORCE_DOCKER_BUILD:-0}"
RUN_ROSDEP="${RUN_ROSDEP:-0}"

# ------------------------------------------------------------
# Check host src folder
# ------------------------------------------------------------
if [ ! -d "${HOST_SRC}" ]; then
    echo "[ERROR] Could not find host ROS src directory:"
    echo "        ${HOST_SRC}"
    echo ""
    echo "Your expected structure should be:"
    echo "  HUSKY_VOICE_MODULE/"
    echo "  ├── src/"
    echo "  └── docker/ros_noetic_docker/pc/run.sh"
    exit 1
fi

echo "[INFO] Docker folder:"
echo "       ${SCRIPT_DIR}"
echo "[INFO] Host ROS src:"
echo "       ${HOST_SRC}"
echo "[INFO] Container ROS src:"
echo "       ${CONTAINER_SRC}"

# ------------------------------------------------------------
# Build image if missing or forced
# ------------------------------------------------------------
if [ "${FORCE_DOCKER_BUILD}" = "1" ] || ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "[INFO] Building Docker image: ${IMAGE_NAME}"
    echo "[INFO] ROS variant: ${ROS_VARIANT}"

    docker build \
        --build-arg ROS_VARIANT="${ROS_VARIANT}" \
        -t "${IMAGE_NAME}" \
        "${SCRIPT_DIR}"
else
    echo "[INFO] Docker image already exists: ${IMAGE_NAME}"
fi

# ------------------------------------------------------------
# X11 GUI support
# ------------------------------------------------------------
if command -v xhost >/dev/null 2>&1; then
    echo "[INFO] Allowing Docker root user to access X11 display"
    xhost +local:root >/dev/null
fi

XAUTHORITY_FILE="${XAUTHORITY:-$HOME/.Xauthority}"

XAUTH_ARGS=()
if [ -f "${XAUTHORITY_FILE}" ]; then
    XAUTH_ARGS+=("-v" "${XAUTHORITY_FILE}:/root/.Xauthority:ro")
    XAUTH_ARGS+=("-e" "XAUTHORITY=/root/.Xauthority")
fi

# ------------------------------------------------------------
# Audio / microphone support
# Useful for USB microphone, ALSA, PulseAudio
# ------------------------------------------------------------
AUDIO_ARGS=()

if [ -d /dev/snd ]; then
    AUDIO_ARGS+=("-v" "/dev/snd:/dev/snd")
    AUDIO_ARGS+=("--group-add" "audio")
fi

if [ -n "${XDG_RUNTIME_DIR:-}" ] && [ -d "${XDG_RUNTIME_DIR}/pulse" ]; then
    AUDIO_ARGS+=("-v" "${XDG_RUNTIME_DIR}/pulse:${XDG_RUNTIME_DIR}/pulse")
    AUDIO_ARGS+=("-e" "PULSE_SERVER=unix:${XDG_RUNTIME_DIR}/pulse/native")
fi

# ------------------------------------------------------------
# Remove previous container with same name
# The build/devel/log volumes remain, so rebuilds are faster.
# ------------------------------------------------------------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] Removing old container: ${CONTAINER_NAME}"
    docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

# ------------------------------------------------------------
# Run container
# ------------------------------------------------------------
docker run -it \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --privileged \
    --net=host \
    --ipc=host \
    --device-cgroup-rule='c 189:* rmw' \
    -e DISPLAY="${DISPLAY}" \
    -e QT_X11_NO_MITSHM=1 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}" \
    -e ROS_IP="${ROS_IP:-}" \
    -e ROS_HOSTNAME="${ROS_HOSTNAME:-}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    "${XAUTH_ARGS[@]}" \
    "${AUDIO_ARGS[@]}" \
    -v /dev:/dev \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /run/udev:/run/udev:ro \
    -v "${HOST_SRC}:${CONTAINER_SRC}:rw" \
    -v "${CONTAINER_NAME}_build:${CONTAINER_WS}/build" \
    -v "${CONTAINER_NAME}_devel:${CONTAINER_WS}/devel" \
    -v "${CONTAINER_NAME}_logs:${CONTAINER_WS}/logs" \
    -w "${CONTAINER_WS}" \
    "${IMAGE_NAME}" \
    bash -lc "
        set -e

        echo ''
        echo '[INFO] Container started.'
        echo '[INFO] GPU check:'
        nvidia-smi || true

        echo ''
        echo '[INFO] USB check:'
        lsusb || true

        echo ''
        echo '[INFO] ROS environment:'
        echo 'ROS_DISTRO=' \$ROS_DISTRO
        echo 'ROS_MASTER_URI=' \$ROS_MASTER_URI
        echo 'ROS_IP=' \$ROS_IP
        echo 'ROS_HOSTNAME=' \$ROS_HOSTNAME

        source /opt/ros/noetic/setup.bash
        cd ${CONTAINER_WS}

        echo ''
        echo '[INFO] Preparing catkin workspace...'

        if [ ! -f ${CONTAINER_WS}/.catkin_tools/profiles/default/config.yaml ]; then
            catkin init
            catkin config --extend /opt/ros/noetic
        fi

        if [ '${RUN_ROSDEP}' = '1' ]; then
            echo ''
            echo '[INFO] Running rosdep install...'
            rosdep update --rosdistro noetic || true
            rosdep install --from-paths src --ignore-src -r -y --rosdistro noetic || true
        fi

        echo ''
        echo '[INFO] Building mounted ROS packages...'
        catkin build

        echo ''
        echo '[INFO] Sourcing workspace...'
        source ${CONTAINER_WS}/devel/setup.bash

        echo ''
        echo '[INFO] Ready.'
        echo '[INFO] Host folder mounted into container:'
        echo '       ${HOST_SRC}  ->  ${CONTAINER_SRC}'
        echo ''
        echo '[INFO] Any edit inside /catkin_ws/src is saved directly on your host.'
        echo ''

        exec terminator
    "

