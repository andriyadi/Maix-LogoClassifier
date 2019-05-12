# Maix-LogoClassifier

A simple logo classifier developed using Maixduino framework and PlatfomIO, to run on K210 MCU on Sipeed's Maix dev board. 
I trained my own ML model, using transfer learning from MobileNet v1. 

## Demo video
Click the thumbnail

[![Demo video thumbnail](https://img.youtube.com/vi/GvPS3iD2f5A/hqdefault.jpg)](https://www.youtube.com/watch?v=GvPS3iD2f5A)

## Prerequisites
* [PlatformIO](http://platformio.org/)
* [platform-kendryte210](https://github.com/sipeed/platform-kendryte210). Should be installed automatically
* Kendryte `nncase` for NeuralNet optimization, download from [here](https://github.com/kendryte/nncase). Unzip anywhere.
* If you're like me, I'll use VSCode and install PlatformIO extension. Maixduino is available for Arduino IDE, but real programmer knows what they should use.

## Train your model
* Install Tensorflow, Keras, and other stuffs. RTFM.
* As the trained model leverages MobileNet, apparently we need to adjust it to be compatible with K210. Replace `mobilenet.py` file on `site-packages/keras_applications` (don't forget to backup) with the one in this repo. `site-packages` folder may exist on several places depends on your environment. If you use **virtualenv**, it should be under `you_virtualenv_dir/lib/python3.x`
* Take a look at [`training/mbnet_keras.py`](https://github.com/andriyadi/Maix-LogoClassifier/blob/master/training/mbnet_keras.py) file. Adjust the constants, and run it.
* Convert the generated `h5` model file by running `training/convert.sh` script with the h5 model file as parameter. Eg. `./convert.sh logoclassifier.h5`
* Copy the generated kmodel file to `src`
* Adjust the labels on `src/names.cpp` file

(More complete steps will be coming soon)

## Credit
* Some code and steps are inspired by this [useful tutorial](https://www.instructables.com/id/Transfer-Learning-With-Sipeed-MaiX-and-Arduino-IDE/). Thanks for your support @AIWintermuteAI 
* MobileNet class is adapted from MBNet_1000 class from [Maixduino](http://github.com/sipeed/Maixduino) example
