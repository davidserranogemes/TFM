# Use an official Python runtime as a parent image
# Desbloquear la adecuada
FROM python:2.7-slim
#FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install request
RUN pip install tabulate
RUN pip install "colorama>=0.3.8"
RUN pip install future

RUN pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o


