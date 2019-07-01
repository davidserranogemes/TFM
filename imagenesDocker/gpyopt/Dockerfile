# Use an official Python runtimefind '/var/lib/docker/volumes/' -mindepth 1 -maxdepth 1 -type d | grep -vFf <(
  docker ps -aq | xargs docker inspect | jq -r '.[] | .Mounts | .[] | .Name | select(.)'
) | xargs -r rm -fr as a parent image
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

RUN apt-get install -y gcc

RUN git clone https://github.com/SheffieldML/GPy
RUN find GPy -name '*.pyx' -exec cython {} \;
RUN pip install Gpy/


RUN pip install gpyopt


