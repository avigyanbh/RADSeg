# Dockerfile to run rayfronts on an Nvidia jetson platform.
# Run with --gpus all --network host --ipc host --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all 

FROM nvcr.io/nvidia/l4t-jetpack:r36.3.0
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y

## Install ROS2 Humble
RUN apt update && apt install locales && locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 && export LANG=en_US.UTF-8

RUN apt install software-properties-common -y && add-apt-repository universe && \
 apt update && apt install curl -y && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && apt update
 
 RUN apt install ros-humble-desktop -y

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Setup python utils
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-requests \
    python3-yaml


## Install PyTorch

# We install prebuilt PyTorch wheels for Jetson provided here
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
# As the base l4t image that has pytorch does not support l4t36.3 yet.
RUN apt-get update && apt-get install -y libopenblas-dev
ARG PYTORCH_URL=https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl
ARG PYTORCH_WHL=torch-2.3.0-cp310-cp310-linux_aarch64.whl
RUN wget -nv -O ${PYTORCH_WHL} ${PYTORCH_URL} && \
    python3 -m pip install --no-cache-dir ${PYTORCH_WHL} && \
    rm ${PYTORCH_WHL}
ARG TORCHAUDIO_URL=https://nvidia.box.com/shared/static/9agsjfee0my4sxckdpuk9x9gt8agvjje.whl
ARG TORCHAUDIO_WHL=torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
RUN wget -nv -O ${TORCHAUDIO_WHL} ${TORCHAUDIO_URL} && \
    python3 -m pip install --no-cache-dir ${TORCHAUDIO_WHL} && \
    rm ${TORCHAUDIO_WHL}
ARG TORCHVISION_URL=https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl
ARG TORCHVISION_WHL=torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
RUN wget -nv -O ${TORCHVISION_WHL} ${TORCHVISION_URL} && \
    python3 -m pip install --no-cache-dir ${TORCHVISION_WHL} && \
    rm ${TORCHVISION_WHL}

## Install python dependencies
RUN pip install \
  protobuf \
  onnx \
  scipy==1.15.2 \
  rerun-sdk==0.22.0 \
  einops \
  timm \
  torch-scatter==2.1.2 \
  ftfy \
  regex \
  nanobind \
  hydra-core \
  open_clip_torch \
  transformers \
  idna==3.10 \
  requests==2.32.3 \
  pandas

## Compile & install patched open-vdb (Need open-vdb 12.0 that exposes Int8Grid to python)

# Get > 1.8 boost version for compatibility with openvdb 12.0 
# But need to downgrade numpy as latest boost 1.86 not compatible with numpy 2
RUN pip install "numpy<2"
RUN wget https://archives.boost.io/release/1.86.0/source/boost_1_86_0.tar.gz && \
  tar xzvf boost_1_86_0.tar.gz && cd boost_1_86_0 && ./bootstrap.sh --prefix=/usr && ./b2 && ./b2 install


WORKDIR /workspace
RUN apt-get update && apt-get install -y git && \ 
rm -rf /var/lib/apt/lists/* && \ 
git clone https://github.com/OasisArtisan/openvdb && mkdir openvdb/build

WORKDIR /workspace/openvdb/build
RUN apt-get install -y cmake libboost-iostreams-dev libtbb-dev libblosc-dev     python3-dev python3-numpy python3-pip
RUN pip3 install nanobind
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr/local \
-DOPENVDB_BUILD_PYTHON_MODULE=ON \
-DOPENVDB_BUILD_PYTHON_UNITTESTS=ON \
-DOPENVDB_PYTHON_WRAP_ALL_GRID_TYPES=ON \
-DUSE_NUMPY=ON \
-Dnanobind_DIR=/usr/local/lib/python3.10/dist-packages/nanobind/cmake ..  
 
RUN make -j4
RUN make install

# Clone rayfronts and compile
WORKDIR /workspace
RUN git clone https://github.com/RayFronts/RayFronts.git
RUN cd RayFronts && CMAKE_INSTALL_PREFIX=/usr/local ./compile.sh

WORKDIR /
