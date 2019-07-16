#!/bin/bash

DATE=$(echo `date +%Y-%m-%d`)



AUTOKERAS_CPU_CNN=false
AUTOKERAS_CPU_MLP=true

AUTOKERAS_GPU_CNN=false
AUTOKERAS_GPU_MLP=true

AUTOKERAS_GPU_CNN_MOD=false
AUTOKERAS_GPU_MLP_MOD=false



cd
cd master/TFM

ALGORITHM="AUTOKERAS"

mkdir "logs/$ALGORITHM"
mkdir "logs/$ALGORITHM/$DATE"

echo "PRUEBAS AUTOKERAS"

if $AUTOKERAS_CPU_CNN; then
	echo "Launch Autokeras CPU CNN "
	mkdir "logs/$ALGORITHM/$DATE/AUTOKERAS_CPU_CNN"
	MODE="AUTOKERAS_CPU_CNN"

	echo "Launch mnist Convolutional CPU"
	docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py mnist Convolutional CPU NOMOD> logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	echo "Launch fashion Convolutional CPU"
	docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py fashion Convolutional CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	echo "Launch cifar10 Convolutional CPU"
	#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py cifar10 Convolutional CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/cifar10.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/cifar10.txt &


	#Clean the system
	echo "Limpìando los residuos generados por docker...."
	docker system prune -f

	echo "Subiendo los resultados a github"
	git pull --force
	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push
	
fi

if $AUTOKERAS_CPU_MLP; then
	echo "Launch Autokeras CPU MLP "
	mkdir "logs/$ALGORITHM/$DATE/AUTOKERAS_CPU_MLP"
	MODE="AUTOKERAS_CPU_MLP"

	#echo "Launch mnist Feedforward CPU"
	#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py mnist Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	#echo "Launch fashion Feedforward CPU"
	#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py fashion Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	echo "Launch imdb Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py imdb Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/imdb.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/imdb.txt &

	echo "Launch letters Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py letters Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/letters.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/letters.txt &



	echo "Launch cifar10 Feedforward CPU"
	#docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py cifar10 Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/cifar10.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/cifar10.txt &


	#Clean the system
	echo "Limpìando los residuos generados por docker...."
	docker system prune -f

	echo "Subiendo los resultados a github"
	git pull --force
	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push

fi



if $AUTOKERAS_GPU_CNN; then
	echo "Launch Autokeras GPU CNN "
	mkdir "logs/$ALGORITHM/$DATE/AUTOKERAS_GPU_CNN"
	MODE="AUTOKERAS_GPU_CNN"

	echo "Launch Autokeras GPU CNN"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate autokeras-gpu


	echo "Launch mnist Convolutional GPU"
	python ficherosEjecuciones/ejecuciones_autokeras.py mnist Convolutional GPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	echo "Launch fashion Convolutional GPU"
	python ficherosEjecuciones/ejecuciones_autokeras.py fashion Convolutional GPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	echo "Launch cifar10 Convolutional GPU"
	#python ficherosEjecuciones/ejecuciones_autokeras.py cifar10 Convolutional GPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/cifar10.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/cifar10.txt &

	conda deactivate

	echo "Subiendo los resultados a github"
	git pull --force
	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push

fi


if $AUTOKERAS_GPU_MLP; then
	echo "Launch Autokeras GPU MLP "
	mkdir "logs/$ALGORITHM/$DATE/AUTOKERAS_GPU_MLP"
	MODE="AUTOKERAS_GPU_MLP"

	echo "Launch Autokeras GPU MLP"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate autokeras-gpu


	#echo "Launch mnist Feedforward GPU"
	#python ficherosEjecuciones/ejecuciones_autokeras.py mnist Feedforward GPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	#echo "Launch fashion Feedforward GPU"
	#python ficherosEjecuciones/ejecuciones_autokeras.py fashion Feedforward GPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	#echo "Launch cifar10 Feedforward GPU"
	#python ficherosEjecuciones/ejecuciones_autokeras.py cifar10 Feedforward GPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/cifar10.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/cifar10.txt &


	echo "Launch imdb Feedforward CPU"
	python ficherosEjecuciones/ejecuciones_autokeras.py imdb Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/imdb.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/imdb.txt &

	echo "Launch letters Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py letters Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/letters.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/letters.txt &


	conda deactivate

	echo "Subiendo los resultados a github"
	git pull --force
	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push

fi





if $AUTOKERAS_GPU_CNN_MOD; then
	echo "Launch Autokeras GPU CNN Modified "
	mkdir "logs/$ALGORITHM/$DATE/AUTOKERAS_GPU_CNN_MOD"
	MODE="AUTOKERAS_GPU_CNN_MOD"

	echo "Launch Autokeras GPU"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate autokeras-modified-gpu


	echo "Launch mnist Convolutional GPU MOD"
	python ficherosEjecuciones/ejecuciones_autokeras_mod.py mnist Convolutional GPU MOD> logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	./acc_autokeras_extractor_mod.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	echo "Launch fashion Convolutional GPU MOD"
	python ficherosEjecuciones/ejecuciones_autokeras_mod.py fashion Convolutional GPU MOD > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	./acc_autokeras_extractor_mod.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	echo "Launch cifar10 Convolutional GPU MOD"
	#python ficherosEjecuciones/ejecuciones_autokeras_mod.py cifar10 Convolutional GPU MOD > logs/$ALGORITHM/$DATE/$MODE/cifar10.txt
	#./acc_autokeras_extractor_mod.sh logs/$ALGORITHM/$DATE/$MODE/cifar10.txt &

	conda deactivate
	
	echo "Subiendo los resultados a github"
	git pull --force

	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push

fi


if $AUTOKERAS_GPU_MLP_MOD; then
	echo "Launch Autokeras GPU MLP Modified"
	mkdir "logs/$ALGORITHM/$DATE/AUTOKERAS_GPU_MLP_MOD"
	MODE="AUTOKERAS_GPU_MLP_MOD"

	echo "Launch Autokeras GPU"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate autokeras-modified-gpu


	echo "Launch mnist Feedforward GPU MOD"
	python ficherosEjecuciones/ejecuciones_autokeras_mod.py mnist Feedforward GPU MOD > logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	echo "Launch fashion Feedforward GPU MOD"
	python ficherosEjecuciones/ejecuciones_autokeras_mod.py fashion Feedforward GPU MOD > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	echo "Launch cifar10 Feedforward GPU MOD"
	#python ficherosEjecuciones/ejecuciones_autokeras_mod.py cifar10 Feedforward GPU MOD > logs/$ALGORITHM/$DATE/$MODE/cifar10.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/cifar10.txt &

	conda deactivate

	echo "Subiendo los resultados a github"
	git pull --force

	git add logs/$ALGORITHM/$DATE/$MODE/*
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push

fi



#Aparagamos el ordenador
poweroff