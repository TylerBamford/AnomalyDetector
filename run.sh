#!/bin/sh

unzip test-dataset.csv.zip -d TestingData
unzip test-dataset.csv.zip -d TrainingData
cp metadata.json TrainingData/metadata.json

docker build -t anomaly-detector . && docker system prune -f
docker-compose up && docker system prune -f