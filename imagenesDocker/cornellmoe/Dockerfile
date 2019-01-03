# Use an official Python runtime as a parent image
# Desbloquear la adecuada
FROM python:2.7-slim
#FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y python-dev gcc cmake libboost-all-dev python-pip libblas-dev liblapack-dev gfortran git python-numpy python-scipy


RUN git clone https://github.com/wujian16/Cornell-MOE.git
RUN cd Cornell-MOE
RUN pip install -r requirements.txt
RUN python setup.py install

RUN pip install 


