FROM ubutu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake ninja-build wget ca-certificates && rm -rf /var/lib/apt/lists/*

ARG NS3_VERSION=3.48
RUN wget -q https://www.nsnam.org/releases/ns-allinone-${NS3_VERSION}.tar.bz2 \
    && tar xf ns-allinone-${NS3_VERSION}.tar.bz2
WORKDIR /ns-allinone-${NS3_VERSION}/ns-${NS3_VERSION}

RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/ns3 \
    -DNS3_EXAMPLES=OFF -DNS3_TESTS=OFF -DNS3_PYTHON_BINDINGS=OFF \
    -DNS3_ENABLED_MODULES="olsr;wifi;mobility;internet;applications;propagation;netanim" \
    && cmake --build build && cmake --install build

COPY simulation/ns3/ /src/ns3/
RUN cmake -B /src/ns3/build -S /src/ns3 -DCMAKE_BUILD_TYPE=Release -DNS3_PREFIX=/opt/ns3 \
    && cmake --build /src/ns3/build

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3-pip python3.12-venv && rm -rf /var/lib/apt/lists/*

COPY --from=ns3-build /opt/ns3/lib /opt/ns3/lib
COPY --from=ns3-build /src/ns3/build/uav-olsr-dataset /app/simulation/ns3/build/uav-olsr-dataset
ENV LD_LIBRARY_PATH=/opt/ns3/lib

WORKDIR /app
COPY requirements.txt .
RUN python3.12 -m venv /venv && /venv/bin/pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt dvc
ENV PATH="/venv/bin:$PATH"

COPY . .
CMD ["bash"]
