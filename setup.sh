#!/bin/bash

pip install keras || exit 1
#pip install tensorflow || exit 1
pip install tensorflow-gpu==1.14.0
pip install pillow || exit 1
pip install numpy || exit 1
pip install scipy || exit 1
pip install imageio || exit 1
pip install matplotlib || exit 1