FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

## The MAINTAINER instruction sets the author field of the generated images.
MAINTAINER am2234@cam.ac.uk

## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get -y install gcc && apt-get -y install g++ &&  apt-get -y install gfortran
RUN apt-get -y install git

## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
