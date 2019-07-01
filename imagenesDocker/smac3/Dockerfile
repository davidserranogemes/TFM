# Use an official Python runtime as a parent image
# Desbloquear la adecuada
#FROM python:2.7-slim
FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get upgrade -y 

RUN apt-get install -y gcc
RUN apt-get install -y g++

RUN apt-get install -y swig



RUN pip install setuptools
RUN pip install cython
RUN pip install numpy>=1.7.1
RUN pip install scipy>=0.18.1
RUN pip install six
RUN pip install psutil
RUN pip install pynisher>=0.4.1
RUN pip install ConfigSpace>=0.4.6
RUN pip install scikit-learn>=0.18.0
RUN pip install pyrfr>=0.5
RUN pip install sphinx
RUN pip install sphinx_rtd_theme
RUN pip install joblib
RUN pip install nose>=1.3.0
RUN pip install pyDOE
RUN pip install sobol_seq
RUN pip install statsmodels
RUN pip install emcee>=2.1.0
RUN pip install pybind11
RUN pip install george


RUN pip install smac
