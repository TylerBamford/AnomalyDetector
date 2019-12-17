FROM centos:7

# Install dependencies
RUN yum install -y epel-release
RUN yum install -y python36 python36-pip
RUN pip3 install tensorflow
RUN pip3 install keras
RUN pip3 install pandas
RUN pip3 install matplotlib

# Declare mount points
VOLUME /anomaly-detector/NetworkModels
VOLUME /anomaly-detector/plots
VOLUME /anomaly-detector/TestingData
VOLUME /anomaly-detector/TrainingData

# Start the AnomalyDetector
ENV PYTHONUNBUFFERED=1
ENTRYPOINT python3 AnomalyDetector.py

# Copy source code
WORKDIR /anomaly-detector
COPY . .