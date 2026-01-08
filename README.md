## Download and Train All

You can run the bash script

	run_all.sh

To both download and train all the datasets in the Roboflow account.

## Download All

To download all of the datasets in the Roboflow account:

	python download.py

These datasets are ready to be trained by the below commands

## Train All

To train all of the downloaded datasets:

	python train_all.py

## Train Individual

To train, open directory in terminal and type

python train.py --img 640 --epochs 3000 --data data/data_file.yaml --weights weight_file.pt

for example:

python train.py --img 640 --epochs 3000 --data data/laserab.yaml --weights yolov5s.pt
