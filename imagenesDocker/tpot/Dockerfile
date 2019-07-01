# Use an official Python runtime as a parent image
# Desbloquear la adecuada
#FROM python:2.7-slim
FROM python:3.6-slim

RUN mkdir /app

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install numpy
RUN pip install scipy
RUN pip install scikit-learn
RUN pip install deap
RUN pip install update_checker
RUN pip install tqdm
RUN pip install stopit
RUN pip install pandas
RUN pip install xgboost
RUN pip install dask[delayed] dask-ml
RUN pip install scikit-mdr skrebate


RUN pip install tpot



