init:
	echo "Initialization..."
	mkdir data/
	mkdir models/

# VARIABLES
# DATA_DIR="/home/joseph/dev/tensorflow/tensorflow2/flower_photos/"
# echo DATA_DIR

reset:
	echo "Initialization..."
	rm -r data/*.csv
	rm -r models/

all: train encode build

train:
	echo "Training model..."
	python train.py \
		--data_dir="/home/joseph/dev/tensorflow/tensorflow2/flower_photos/" \
		--input=150 \
		--tensorboard_path="logs/" \
		--checkpoint_dir="models/ckpt/" \
		--save_dir="models/"
	echo "Done !"

encode:
	echo "Compressing images..."
	python encode.py \
		--data_dir="/home/joseph/dev/tensorflow/tensorflow2/flower_photos/" \
		--model_path="models/" \
		--input=150 \
		--output="data/features.npy"
	echo "Done !"

build:
	echo "Building ANNOY Graph..."
	python build.py \
		--input="data/features.npy" \
		--size=2048 \
		--tree=100 \
		--output="models/"
	echo "Done !"

deploy:
	sh serving/deploy.sh
