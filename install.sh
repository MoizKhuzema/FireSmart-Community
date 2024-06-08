#!/bin/bash

pip --version || sudo apt install python3-pip 
sudo apt install python3-venv
python3 -m ensurepip --upgrade


ENV_DIR="env/"
if [ ! -d "$ENV_DIR" ]; then
    python3 -m venv env
fi

source env/bin/activate



pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
pip3 install tensorflow
gdalinfo --version || sudo apt install gdal-bin
gdalinfo --version || sudo apt install libgdal-dev


# Install all basic requirements
pip3 install -r requirements.txt




# Install GDAL

GDAL_MSG=$(gdalinfo --version)
GDAL_MSG_ARRAY=($GDAL_MSG)
GDAL_VER=$(echo "${GDAL_MSG_ARRAY[1]}" | sed 's/,//g')
pip3 install GDAL==$GDAL_VER
