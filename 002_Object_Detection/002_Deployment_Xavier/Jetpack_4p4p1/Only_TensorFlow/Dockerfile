FROM nvcr.io/nvidia/l4t-ml:r32.4.3-py3
WORKDIR /
RUN apt update && apt install -y --fix-missing make g++ pkg-config
RUN apt update && apt install -y --fix-missing python3-pip hdf5-tools protobuf-compiler
RUN apt update && apt install -y libhdf5-serial-dev libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/lib/aarch64-linux-gnu/hdf5/serial
RUN pip3 install -U pip Cython
RUN pip3 install -U future==0.17.1 mock==3.0.5 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 numpy==1.16.1  enum34 futures protobuf pybind11
RUN pip3 install -U h5py
RUN pip3 install -U pip testresources setuptools


RUN pip3 install https://developer.download.nvidia.com/compute/redist/jp/v43/tensorflow-gpu/tensorflow_gpu-1.15.0+nv19.12-cp36-cp36m-linux_aarch64.whl
 
RUN apt install -y build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev

RUN git clone https://github.com/opencv/opencv.git
WORKDIR /opencv
RUN git checkout 4.4.1
WORKDIR /opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local ..

ENV PYTHONPATH $PYTHONPATH:/ml/models/research/:/ml/models/research/slim
#ENV PYTHON3_EXECUTABLE /usr/bin/python3.6
#ENV PYTHON3_EXECUTABLE /usr/bin/python3.6
#ENV PYTHON_INCLUDE_DIR2  /usr/include/aarch64-linux-gnu/python3.6m
#ENV PYTHON_LIBRARY  /usr/lib/aarch64-linux-gnu/libpython3.6m.so
#ENV PYTHON3_NUMPY_INCLUDE_DIRS  /usr/lib/python3/dist-packages/numpy/core/include/

RUN make -j4
RUN make install


RUN pip3 install matplotlib pillow

#COPY ML /ml

#WORKDIR /ml


CMD ["/bin/bash"]

#CMD ["python3","server_ml.py" ]
