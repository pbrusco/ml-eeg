#!/bin/sh

python setup.py bdist_wheel --universal
python setup.py bdist_wheel sdist upload
sudo pip2 install --upgrade ml-eeg
sudo pip3 install --upgrade ml-eeg
