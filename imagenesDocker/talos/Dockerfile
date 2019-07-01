# Use an official Python runtime as a parent image
# Desbloquear la adecuada
FROM davidserranogemes/keras-tensorflow1.8-no-avx-python-2.7
#FROM python:3.6-slim

#RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

#Neceista compilador de C
RUN apt-get update && \
    apt-get upgrade -y 
Run apt-get install -y gcc

RUN apt-get update && apt-get install -y tk-dev && rm -r /var/lib/apt/lists/* 

# Install any needed packages specified in requirements.txt
RUN pip install requests 
RUN pip install  talos


