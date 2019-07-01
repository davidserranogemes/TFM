# Use an official Python runtime as a parent image
# Desbloquear la adecuada
#FROM python:2.7-slim
FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app


#Neceista compilador de C
#RUN apt-get update && \
#    apt-get upgrade -y 
#Run apt-get install -y gcc
# Install any needed packages specified in requirements.txt
#Es necesario instalarlo desde el source
#pip install -U  pip six numpy wheel mock
#pip install -U  keras_applications==1.0.6 --no-deps
#pip install -U  keras_preprocessing==1.0.5 --no-deps

#RUN apt-get install -y pkg-config zip g++ zlib1g-dev unzip python 

#RUN chmod +x bazel-0.22.0-installer-linux-x86_64.sh
#RUN /bazel-0.22.0-installer-linux-x86_64.sh --user
#RUN export PATH="$PATH:$HOME/bin"


#RUN git clone https://github.com/tensorflow/tensorflow.git
#RUN cd tensorflow
#RUN ./configure

#RUN bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package --local_resources 4096 --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0"

#RUN ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git


RUN pip install tensorflow
RUN pip install keras


