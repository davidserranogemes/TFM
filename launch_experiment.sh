#!/bin/bash

DATE=$(echo `date +%Y-%m-%d`)







cd
cd master/TFM


##############################################################################################################
##############################################################################################################
#########																							##########
#########																							##########
#########											AUTOKERAS										##########
#########																							##########
#########																							##########
##############################################################################################################
##############################################################################################################

AUTOKERAS_CPU_CNN=false
AUTOKERAS_CPU_MLP=false

AUTOKERAS_GPU_CNN=false
#AUTOKERAS_GPU_MLP=true
AUTOKERAS_GPU_MLP=false

AUTOKERAS_GPU_CNN_MOD=false






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

	echo "Launch mnist Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py mnist Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	echo "Launch fashion Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py fashion Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	echo "Launch imdb Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py imdb Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/imdb.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/imdb.txt &

	echo "Launch letters Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/autokeras python ficherosEjecuciones/ejecuciones_autokeras.py letters Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/letters.txt
	./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/letters.txt &


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

	#echo "Launch imdb Feedforward CPU"
	#python ficherosEjecuciones/ejecuciones_autokeras.py imdb Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/imdb.txt
	#./acc_autokeras_extractor.sh logs/$ALGORITHM/$DATE/$MODE/imdb.txt &

	echo "Launch letters Feedforward CPU"
	python ficherosEjecuciones/ejecuciones_autokeras.py letters Feedforward CPU NOMOD > logs/$ALGORITHM/$DATE/$MODE/letters.txt
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

	conda deactivate
	
	echo "Subiendo los resultados a github"
	git pull --force

	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push

fi


##############################################################################################################
##############################################################################################################
#########																							##########
#########																							##########
#########											HYPERAS											##########
#########																							##########
#########																							##########
##############################################################################################################
##############################################################################################################


HYPERAS_CPU_CNN=true
HYPERAS_CPU_MLP=false

HYPERAS_GPU_CNN=false
HYPERAS_GPU_MLP=false


ALGORITHM="HYPERAS"

mkdir "logs/$ALGORITHM"
mkdir "logs/$ALGORITHM/$DATE"

echo "PRUEBAS HYPERAS"

if $HYPERAS_CPU_CNN; then
	echo "Launch Hyperas CPU CNN "
	mkdir "logs/$ALGORITHM/$DATE/HYPERAS_CPU_CNN"
	MODE="HYPERAS_CPU_CNN"

	echo "Launch mnist Convolutional CPU"
	docker run -v"$(pwd)":/app davidserranogemes/hyperas python ficherosEjecuciones/ejecuciones_hyperas.py mnist Convolutional CPU > logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	echo "Launch fashion Convolutional CPU"
	docker run -v"$(pwd)":/app davidserranogemes/hyperas python ficherosEjecuciones/ejecuciones_hyperas.py fashion Convolutional CPU  > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &



	#Clean the system
	echo "Limpìando los residuos generados por docker...."
	docker system prune -f

	echo "Subiendo los resultados a github"
	git pull --force
	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push
	
fi

if $HYPERAS_CPU_MLP; then
	echo "Launch Hyperas CPU MLP "
	mkdir "logs/$ALGORITHM/$DATE/HYPERAS_CPU_MLP"
	MODE="HYPERAS_CPU_MLP"

	echo "Launch mnist Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/hyperas python ficherosEjecuciones/ejecuciones_hyperas.py mnist Feedforward CPU  > logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	echo "Launch fashion Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/hyperas python ficherosEjecuciones/ejecuciones_hyperas.py fashion Feedforward CPU  > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	echo "Launch imdb Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/hyperas python ficherosEjecuciones/ejecuciones_hyperas.py imdb Feedforward CPU  > logs/$ALGORITHM/$DATE/$MODE/imdb.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/imdb.txt &

	echo "Launch letters Feedforward CPU"
	docker run -v"$(pwd)":/app davidserranogemes/hyperas python ficherosEjecuciones/ejecuciones_hyperas.py letters Feedforward CPU  > logs/$ALGORITHM/$DATE/$MODE/letters.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/letters.txt &


	#Clean the system
	echo "Limpìando los residuos generados por docker...."
	docker system prune -f

	echo "Subiendo los resultados a github"
	git pull --force
	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push

fi



if $HYPERAS_GPU_CNN; then
	echo "Launch Hyperas GPU CNN "
	mkdir "logs/$ALGORITHM/$DATE/HYPERAS_GPU_CNN"
	MODE="HYPERAS_GPU_CNN"

	echo "Launch Hyperas GPU CNN"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate hyperas-gpu


	echo "Launch mnist Convolutional GPU"
	python ficherosEjecuciones/ejecuciones_hyperas.py mnist Convolutional GPU  > logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	echo "Launch fashion Convolutional GPU"
	python ficherosEjecuciones/ejecuciones_hyperas.py fashion Convolutional GPU  > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	conda deactivate

	echo "Subiendo los resultados a github"
	git pull --force
	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push

fi


if $HYPERAS_GPU_MLP; then
	echo "Launch Hyperas GPU MLP "
	mkdir "logs/$ALGORITHM/$DATE/HYPERAS_GPU_MLP"
	MODE="HYPERAS_GPU_MLP"

	echo "Launch Hyperas GPU MLP"
	source ~/anaconda3/etc/profile.d/conda.sh
	conda activate hyperas-gpu


	echo "Launch mnist Feedforward GPU"
	python ficherosEjecuciones/ejecuciones_hyperas.py mnist Feedforward GPU  > logs/$ALGORITHM/$DATE/$MODE/mnist.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/mnist.txt &

	echo "Launch fashion Feedforward GPU"
	python ficherosEjecuciones/ejecuciones_hyperas.py fashion Feedforward GPU  > logs/$ALGORITHM/$DATE/$MODE/fashion.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/fashion.txt &

	echo "Launch imdb Feedforward CPU"
	python ficherosEjecuciones/ejecuciones_hyperas.py imdb Feedforward CPU  > logs/$ALGORITHM/$DATE/$MODE/imdb.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/imdb.txt &

	echo "Launch letters Feedforward CPU"
	python ficherosEjecuciones/ejecuciones_hyperas.py letters Feedforward CPU  > logs/$ALGORITHM/$DATE/$MODE/letters.txt
	./acc_hyperas_extractor.sh logs/$ALGORITHM/$DATE/$MODE/letters.txt &


	conda deactivate

	echo "Subiendo los resultados a github"
	git pull --force
	git add logs/$ALGORITHM/$DATE/$MODE/
	git commit -m" Resultados logs/$ALGORITHM/$DATE/$MODE/*"
	git push

fi







#Aparagamos el ordenador
poweroff