# Use an official Python runtime as a parent image
# Desbloquear la adecuada
#FROM davidserranogemes/keras-tensorflow1.12-no-avx-python-3.6
#FROM python:3.6-slim
FROM continuumio/miniconda3

# Set the working directory to /app
WORKDIR /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y gcc
RUN apt-get install -y g++


# Install any needed packages specified in requirements.txt

#RUN pip install tensorflow-gpu
RUN conda install -c anaconda keras-gpu 
RUN pip install autokeras

