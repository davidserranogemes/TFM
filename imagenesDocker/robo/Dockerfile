# Use an official Python runtime as a parent image
# Desbloquear la adecuada
#FROM python:2.7-slim
FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app


# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y gcc
RUN apt-get install -y g++



RUN apt-get install -y libeigen3-dev swig 


# Install any needed packages specified in requirements.txt
RUN git clone https://github.com/automl/RoBO

RUN pip install Cython==0.27.3
RUN pip install numpy>=1.7
RUN pip install emcee>=2.1.0
RUN pip install scipy>=0.13.3
RUN pip install nose
RUN pip install pyrfr
RUN pip install Jinja2
RUN pip install pyyaml
RUN pip install pybind11
RUN pip install git+https://github.com/automl/george.git@development
RUN pip install git+https://github.com/automl/pybnn.git


RUN chmod 777 RoBO/setup.py
RUN python RoBO/setup.py install