#!/bin/bash

FOLDER=data

if [[ $1  =~ ^(sketch|real)$ ]];
then
    echo "Downloading data"
    wget http://csr.bu.edu/ftp/visda/2019/multi-source/$1.zip
    mkdir $FOLDER
    mv $1.zip $FOLDER
    cd $FOLDER
    unzip $1.zip "$1/bird/*"
    unzip $1.zip "$1/dog/*"
    unzip $1.zip "$1/flower/*"
    unzip $1.zip "$1/speedboat/*"
    unzip $1.zip "$1/tiger/*"

    rm -rf $1.zip
    cd $1

    mkdir train
    mkdir train/bird
    mkdir train/dog
    mkdir train/flower
    mkdir train/speedboat
    mkdir train/tiger

    mkdir test
    mkdir test/bird
    mkdir test/dog
    mkdir test/flower
    mkdir test/speedboat
    mkdir test/tiger

    ls bird/      | head -15 | xargs -i mv bird/{}      test/bird
    ls dog/       | head -15 | xargs -i mv dog/{}       test/dog
    ls flower/    | head -15 | xargs -i mv flower/{}    test/flower
    ls speedboat/ | head -15 | xargs -i mv speedboat/{} test/speedboat
    ls tiger/     | head -15 | xargs -i mv tiger/{}     test/tiger

    ls bird/      | xargs -i mv bird/{}      train/bird
    ls dog/       | xargs -i mv dog/{}       train/dog
    ls flower/    | xargs -i mv flower/{}    train/flower
    ls speedboat/ | xargs -i mv speedboat/{} train/speedboat
    ls tiger/     | xargs -i mv tiger/{}     train/tiger

    rm -rf bird dog flower speedboat tiger

else
    echo "$1 is not a valid choice."
fi
