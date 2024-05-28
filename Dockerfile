FROM tensorflow/tensorflow:latest-gpu-py3
RUN apt-get update

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get install -y python3-pip python3-tk
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y git
#RUN apt-get install -y python3-libnvinfer-dev

RUN pip3 install opencv-python
RUN pip3 install matplotlib
RUN pip3 install -U git+https://github.com/microsoft/onnxconverter-common
RUN pip3 install -U git+https://github.com/onnx/keras-onnx
RUN apt install -y  libgl1-mesa-glx
RUN pip3 install pandas
