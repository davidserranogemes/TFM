# Use an official Python runtime as a parent image
# Desbloquear la adecuada
FROM python:2.7-slim
#FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

#Neceista compilador de C
RUN apt-get update && \
    apt-get upgrade -y 
Run apt-get install -y gcc

# Install any needed packages specified in requirements.txt
RUN pip install  talos


