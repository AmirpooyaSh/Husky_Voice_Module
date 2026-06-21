# ------------------------------------------------------------
# CUDA 12.6 devel + Ubuntu 20.04 + ROS Noetic
# Location:
#   HUSKY_VOICE_MODULE/docker/ros_noetic_docker/pc/Dockerfile
# ------------------------------------------------------------

# sudo docker build -t cuda126-noetic:desktop-full .

FROM nvidia/cuda:12.6.3-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV ROS_DISTRO=noetic

# ------------------------------------------------------------
# Timezone
# ------------------------------------------------------------
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && \
    apt-get install -q -y --no-install-recommends tzdata && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Base packages
# ------------------------------------------------------------
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    ca-certificates \
    curl \
    dirmngr \
    gnupg2 \
    lsb-release \
    build-essential \
    git \
    wget \
    nano \
    vim \
    less \
    htop \
    usbutils \
    pciutils \
    udev \
    x11-apps \
    mesa-utils \
    xauth \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# ROS Noetic final snapshot key
# ------------------------------------------------------------
RUN set -eux; \
    key='4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA'; \
    export GNUPGHOME="$(mktemp -d)"; \
    gpg --batch --keyserver keyserver.ubuntu.com --recv-keys "$key"; \
    mkdir -p /usr/share/keyrings; \
    gpg --batch --export "$key" > /usr/share/keyrings/ros1-snapshots-archive-keyring.gpg; \
    gpgconf --kill all; \
    rm -rf "$GNUPGHOME"

RUN echo "deb [ signed-by=/usr/share/keyrings/ros1-snapshots-archive-keyring.gpg ] http://snapshots.ros.org/noetic/final/ubuntu focal main" \
    > /etc/apt/sources.list.d/ros1-snapshots.list

# ------------------------------------------------------------
# ROS core
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-core=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# ROS build tools
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-rosdep \
    python3-rosinstall \
    python3-vcstools \
    python3-catkin-tools \
    python3-osrf-pycommon \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# rosdep
# ------------------------------------------------------------
RUN rosdep init || true && \
    rosdep update --rosdistro $ROS_DISTRO

# ------------------------------------------------------------
# ROS base
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-ros-base=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Optional ROS variant
#
# Build options:
#   --build-arg ROS_VARIANT=ros-base
#   --build-arg ROS_VARIANT=robot
#   --build-arg ROS_VARIANT=perception
#   --build-arg ROS_VARIANT=desktop
#   --build-arg ROS_VARIANT=desktop-full
# ------------------------------------------------------------
ARG ROS_VARIANT=ros-base

RUN set -eux; \
    apt-get update; \
    if [ "$ROS_VARIANT" = "ros-base" ]; then \
        echo "Keeping ros-base"; \
    elif [ "$ROS_VARIANT" = "robot" ]; then \
        apt-get install -y --no-install-recommends \
            ros-noetic-robot=1.5.0-1*; \
    elif [ "$ROS_VARIANT" = "perception" ]; then \
        apt-get install -y --no-install-recommends \
            ros-noetic-perception=1.5.0-1*; \
    elif [ "$ROS_VARIANT" = "desktop" ]; then \
        apt-get install -y --no-install-recommends \
            ros-noetic-robot=1.5.0-1* \
            ros-noetic-desktop=1.5.0-1*; \
    elif [ "$ROS_VARIANT" = "desktop-full" ]; then \
        apt-get install -y --no-install-recommends \
            ros-noetic-robot=1.5.0-1* \
            ros-noetic-perception=1.5.0-1* \
            ros-noetic-desktop=1.5.0-1* \
            ros-noetic-desktop-full=1.5.0-1*; \
    else \
        echo "Unknown ROS_VARIANT: $ROS_VARIANT"; \
        exit 1; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Catkin workspace
# Host src will be mounted here by run.sh:
#   HUSKY_VOICE_MODULE/src  ->  /catkin_ws/src
# ------------------------------------------------------------
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin config --extend /opt/ros/noetic"

# ------------------------------------------------------------
# Bash setup
# ------------------------------------------------------------
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "if [ -f /catkin_ws/devel/setup.bash ]; then source /catkin_ws/devel/setup.bash; fi" >> /root/.bashrc && \
    echo "cd /catkin_ws" >> /root/.bashrc

# ------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------
RUN printf '#!/bin/bash\nset -e\nsource /opt/ros/noetic/setup.bash\nif [ -f /catkin_ws/devel/setup.bash ]; then source /catkin_ws/devel/setup.bash; fi\nexec "$@"\n' \
    > /ros_entrypoint.sh && \
    chmod +x /ros_entrypoint.sh

# ------------------------------------------------------------
# Terminator
# ------------------------------------------------------------
RUN apt-get update && apt-get install -q -y --no-install-recommends \
    terminator \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    alsa-utils \
    pulseaudio-utils \
    libasound2-dev \
    python3-requests \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["terminator"]
