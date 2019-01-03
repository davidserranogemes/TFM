# Use an official Python runtime as a parent image
# Desbloquear la adecuada
FROM python:2.7-slim
#FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y libicu-dev
# Install any needed packages specified in requirements.txt
RUN pip install numpy
RUN pip install scipy
RUN pip install -U scikit-learn
RUN pip install PyICU
RUN pip install git+https://github.com/AIworx-Labs/chocolate@master


