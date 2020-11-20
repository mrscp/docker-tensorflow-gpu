FROM tensorflow/tensorflow:latest-gpu

WORKDIR /docker-tf-gpu

#RUN apt-get update && apt-get install -y python3 python3-pip sudo python-scipy
#RUN pip3 install --upgrade pip

#COPY requirements.txt /docker-tf-gpu/
#RUN pip3 install -r requirements.txt

COPY . /docker-tf-gpu/
