# Whisper Inference with TensorFlow Lite
This repository provides a minimal example of running inference on the Whisper ASR model using TensorFlow Lite. The Whisper model is a hybrid model with weights in int8 format and activations in float32 format.

# Getting Started
Follow these steps to run inference with the Whisper model using TensorFlow Lite:

## Step 1: Clone the Repository
Clone the [whisper.tflite](https://github.com/nyadla-sys/whisper.tflite.git) repository. 
Please note that Whisper's model weights use Git Large File Storage (LFS), so you may need to install it using git lfs install if you haven't already.

```bash
git clone https://github.com/nyadla-sys/whisper.tflite.git
cd whisper.tflite/tflite_example
```
## Step 2: Install CMake
Make sure you have CMake version 3.16 or higher installed. On Ubuntu, you can install it with:

```bash
sudo apt-get install cmake
```
Alternatively, you can follow the official CMake installation guide.

## Step 3: Clone TensorFlow Repository
Clone the TensorFlow repository to get the required files.

```bash
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
```
## Step 4: Copy Required Files
Copy the necessary files from this repository to the TensorFlow minimal example directory.

```bash
cp minimal.cc  tensorflow_src/tensorflow/lite/examples/minimal/
cp *.h  tensorflow_src/tensorflow/lite/examples/minimal/
```
## Step 5: Create a CMake Build Directory
Create a build directory for CMake and run the CMake tool.

```bash
mkdir minimal_build
cd minimal_build
cmake ../tensorflow_src/tensorflow/lite/examples/minimal
```
## Step 6: Build TensorFlow Lite
In the minimal_build directory, build the minimal example.

```bash
# Build the minimal example
cmake --build . -j
```
If the CMake build fails, you can try specifying the number of cores with the -j flag, like this:

```bash
cmake --build . -j 8
```

## Step 7: Run Inference
You can now run inference on the Whisper model using pre-generated input features or provide a 16KHz 16-bit mono audio file.

```bash
# Transcribe an audio file
./minimal ../../models/whisper-tiny-en.tflite ../samples/jfk.wav
```

Note: You can use the arecord application on a Linux computer to record test audio files with the following command:

```bash
arecord -r 16000 -c 1 -d 30 -f S16_LE test.wav
```
Now you're all set to run Whisper ASR inference with TensorFlow Lite!
