#!/bin/bash

cd
cd master/TFM
echo "Launch Autokeras "
#echo "Launch mnist Convolutional"
#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py mnist Convolutional > logs/autokeras_mnist_convolutional.txt
#./acc_extractor.sh autokeras_mnist_convolutional.txt &

#echo "Launch fashion Convolutional"
#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py fashion Convolutional > logs/autokeras_fashion_convolutional.txt
#./acc_extractor.sh autokeras_fashion_convolutional.txt &

#echo "Launch cifar10 Convolutional"
#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py cifar10 Convolutional > logs/autokeras_cifar10_convolutional.txt
#./acc_extractor.sh autokeras_cifar10_convolutional.txt &

#Clean the system
#echo "Limpìando los residuos generados por docker...."
#docker system prune -f

#echo "Launch mnist Feedforward"
#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py mnist Feedforward > logs/autokeras_mnist_feedforward.txt
#./acc_extractor.sh autokeras_mnist_feedforward.txt &

#echo "Launch fashion Feedforward"
#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py fashion Feedforward > logs/autokeras_fashion_feedforward.txt
#./acc_extractor.sh autokeras_fashion_feedforward.txt &

#echo "Launch cifar10 Feedforward"
#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py cifar10 Feedforward > logs/autokeras_cifar10_feedforward.txt
#./acc_extractor.sh autokeras_cifar10_feedforward.txt &

#echo "Limpìando los residuos generados por docker...."
#docker system prune -f


echo "Launch Auto_ml"

echo "Launch mnist Feedforward"
docker run -v"$(pwd)":/app davidserranogemes/auto_ml python ficherosEjecuciones/ejecuciones_automl.py mnist Feedforward 

echo "Launch fashion Feedforward"
docker run -v"$(pwd)":/app davidserranogemes/auto_ml python ficherosEjecuciones/ejecuciones_automl.py fashion Feedforward 

echo "Launch cifar10 Feedforward"
docker run -v"$(pwd)":/app davidserranogemes/auto_ml python ficherosEjecuciones/ejecuciones_automl.py cifar10 Feedforward 


echo "Limpìando los residuos generados por docker...."
docker system prune -f

#
