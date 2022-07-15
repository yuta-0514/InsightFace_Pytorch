FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04

COPY requirements.txt /.

RUN apt-get update

RUN apt-get install -y python3-pip

RUN apt-get install -y build-essential libssl-dev libffi-dev python3-dev git

RUN apt-get install -y libgl1-mesa-dev

RUN apt-get autoremove -y &&\
  apt-get clean &&\
  rm -rf /usr/local/src/*

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt --user &&\
  rm -rf ~/.cache/pip
