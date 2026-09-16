FROM osrf/ros:humble-desktop@sha256:fb07245b32187d74350be25323d8ad2f8ca5c25c325759911a1eff2267a49c1e
SHELL ["/bin/bash", "-c"]

ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY

ENV http_proxy=${HTTP_PROXY}
ENV https_proxy=${HTTPS_PROXY}
ENV no_proxy=${NO_PROXY}

ENV DEBIAN_FRONTEND=noninteractive \
    COLCON_WS=/fishbot_ws

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        python3-rosdep \
        python3-vcstool \
        python3-colcon-common-extensions

# DO NOT CLEAR THE `rm -rf /var/lib/apt/lists/*`
# it leads to the errors in rosdep install

# Source ROS and workspace by default
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source $COLCON_WS/install/setup.bash" >> ~/.bashrc
