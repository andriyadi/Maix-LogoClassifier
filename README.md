# Maix-LogoClassifier

A simple logo classifier developed using Maixduino framework and PlatfomIO, to run on K210 MCU on Sipeed's Maix dev board.

## Demo
Click the thumbnail

[![Demo video thumbnail](https://img.youtube.com/vi/GvPS3iD2f5A/hqdefault.jpg)](https://www.youtube.com/watch?v=GvPS3iD2f5A)

## Prerequisites
* [PlatformIO](http://platformio.org/)
* [platform-kendryte210](https://github.com/sipeed/platform-kendryte210). Should be installed automatically
* If you like me, I'll use VSCode and install PlatformIO extension

## Train your model
* Install Tensorflow, Keras, and other stuffs. RTFM.
* Take a look at [`training/mbnet_keras.py`](https://github.com/andriyadi/Maix-LogoClassifier/blob/master/training/mbnet_keras.py) file. Adjust the constants, and run it.
* Convert the generated `h5` model file by running `training/convert.sh` script with the h5 model file as parameter
* Copy the generated kmodel file to `src`


## Credit
* Some code and steps are inspired by this [useful tutorial](https://www.instructables.com/id/Transfer-Learning-With-Sipeed-MaiX-and-Arduino-IDE/). Thanks for your support @AIWintermuteAI 
* MobileNet class is adapted from MBNet_1000 class from [Maixduino](http://github.com/sipeed/Maixduino) example
