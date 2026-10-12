FROM ubuntu:22.04

# Patch Python 3.10 collections to be compatible with Ryu and Eventlet
RUN apt-get update && apt-get install -y python3 && \
    echo "import collections.abc; MutableMapping = collections.abc.MutableMapping; Mapping = collections.abc.Mapping; Iterable = collections.abc.Iterable; Iterator = collections.abc.Iterator; Callable = collections.abc.Callable" >> /usr/lib/python3.10/collections/__init__.py

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
    gymnasium && \
    sed -i 's/base.is_timeout = property(lambda _: True)/pass/g' /usr/local/lib/python3.10/dist-packages/eventlet/timeout.py && \
    python3 -c "p = '/usr/local/lib/python3.10/dist-packages/eventlet/greenio/py3.py'; c = open(p).read().replace('_original_pyio.open', 'getattr(_original_pyio.open, \'__func__\', _original_pyio.open)'); open(p, 'w').write(c)"

WORKDIR /app

COPY sdn/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
