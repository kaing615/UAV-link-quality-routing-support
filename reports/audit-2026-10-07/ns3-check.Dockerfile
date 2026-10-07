FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ cmake ninja-build wget ca-certificates bzip2 python3 \
    && rm -rf /var/lib/apt/lists/*
RUN wget -q https://www.nsnam.org/releases/ns-allinone-3.48.tar.bz2 \
    && tar xf ns-allinone-3.48.tar.bz2
WORKDIR /ns-allinone-3.48/ns-3.48
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/ns3 \
    -DNS3_EXAMPLES=OFF -DNS3_TESTS=OFF -DNS3_PYTHON_BINDINGS=OFF \
    -DNS3_ENABLED_MODULES="olsr;wifi;mobility;internet;applications;propagation;netanim;flow-monitor" \
    && cmake --build build -j 4 && cmake --install build
COPY . /scenario/
RUN cmake -B /scenario/build -S /scenario -G Ninja -DCMAKE_BUILD_TYPE=Release -DNS3_PREFIX=/opt/ns3 \
    && cmake --build /scenario/build \
    && cd /scenario && sha256sum uav-olsr-dataset.cc CMakeLists.txt build/uav-olsr-dataset > build/source.sha256
ENV LD_LIBRARY_PATH=/opt/ns3/lib
WORKDIR /repo
