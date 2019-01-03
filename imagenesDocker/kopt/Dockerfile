# Use an official Python runtime as a parent image
# Desbloquear la adecuada
FROM python:2.7-slim
#FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
# Install hyperopt from the master 
#PAra poder instalar usando pup install git
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y gcc
# Install kopt

RUN pip install git+https://github.com/hyperopt/hyperopt.git
RUN pip install kopt