FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    mininet \
    openvswitch-switch \
    openvswitch-testcontroller \
    iproute2 \
    iputils-ping \
    iperf3 \
    python3 \
    python3-pip \
    curl \
    nano \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip3 install --no-cache-dir \
    eventlet==0.30.2 \
    ryu \
    torch --extra-index-url https://download.pytorch.org/whl/cpu \
    stable-baselines3 \
    numpy \
    networkx \
    scipy \
    gymnasium

WORKDIR /app

COPY sdn/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
