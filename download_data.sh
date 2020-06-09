#!/bin/bash

if [[ $1  =~ ^(sketch|real)$ ]];
then
    echo "Downloading data"
    wget http://csr.bu.edu/ftp/visda/2019/multi-source/$1.zip
    mkdir data
    mv $1.zip data
    cd data
    unzip $1.zip
    mkdir train
    mkdir train/$1
    mv $1/bird train/$1
    mv $1/dog train/$1
    mv $1/flower train/$1
    mv $1/speedboad train/$1
    mv $1/tiger train/$1
    rm -rf $1
    rm -rf $1.zip
else
    echo "$1 is not a valid choice."
fi
