# Download base image for running the environment
FROM ubuntu:20.04

# Update and install the packages
RUN apt-get update -y && apt-get install -y \
    software-properties-common vim
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update -y && apt-get install -y \
    build-essential \
    python3.8 python3.8-dev python3.8-distutils \
    git wget
RUN wget https://bootstrap.pypa.io/get-pip.py && python3.8 get-pip.py && rm get-pip.py
RUN apt-get install -y \
    unzip \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
    libsdl1.2-dev libsdl-image1.2-dev \
    xvfb patchelf ffmpeg cmake swig

# Link the command
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN ln -s /usr/bin/python3.8 /usr/bin/python

# Install the requirements
COPY requirements.txt ./
RUN pip install -r requirements.txt && rm requirements.txt

# Set up home directory
ENV HOME="/root"
WORKDIR ${HOME}

# Install the snake env
COPY snake snake
RUN pip install -e snake

# Copy other resources
COPY src RL_algo
