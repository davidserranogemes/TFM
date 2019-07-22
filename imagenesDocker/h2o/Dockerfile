# Use an official Python runtime as a parent image
# Desbloquear la adecuada
#FROM python:2.7-slim
FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y gcc
RUN apt-get install -y g++

#RUN apt-get install -y software-properties-common

#Install Java
#RUN add-apt-repository -y ppa:webupd8team/java
RUN apt-get update -y

#RUN apt-get install -y openjdk-8-jdk && \

RUN apt-get install openjdk-9-jdk && \
    apt-get install -y ant && \
    apt-get clean;


# Install any needed packages specified in requirements.txt
RUN pip install tensorflow
RUN pip install keras

RUN pip install numpy
RUN pip install SciPy

RUN pip install scikit-learn==0.20



RUN pip install -U request
RUN pip install -U tabulate
RUN pip install -U "colorama>=0.3.8"
RUN pip install -U future

RUN pip install h2o




