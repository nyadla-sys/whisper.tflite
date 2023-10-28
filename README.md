# Whisper Enhanced Quantized TFLite Model

This project contains an enhanced version of the Whisper quantized TFLite model optimized for both Android and iOS platforms. The model is designed to perform well on edge devices, making it suitable for a wide range of applications.

## Integrate Whisper TFLite for Enhanced Mobile App Performance

If you're interested in incorporating the Whisper TFLite model into your iOS and Android applications, please don't hesitate to reach out to us at yadlaniranjan@gmail.com. Our project features an upgraded iteration of the Whisper quantized TFLite model, finely tuned for optimal performance on both Android and iOS platforms. This model is tailored to excel on edge devices, rendering it versatile for various application scenarios. Contact us for further details and collaboration opportunities


## Getting Started

To get started with this enhanced Whisper model, follow these steps:

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/nyadla-sys/whisper.tflite.git
   ```

2. You can now use the enhanced Whisper quantized TFLite model in your projects by refering sample code for Android and iOS.

## Demo Apps

### Android
You can find a sample Android app in the [whisper_android](whisper_android) folder that demonstrates how to use the Whisper TFLite model for transcription on Android devices.

## DTLN quantized tflite model

Our overarching objective is to incorporate real-time noise suppression through the utilization of a quantized [DTLN](https://github.com/breizhn/DTLN) tflite model, delivering noise-reduced audio data to the whisper tflite model.

Courtesy from [breizhn/DTLN](https://github.com/breizhn/DTLN)

[DTLN Paper](https://arxiv.org/pdf/2005.07551.pdf)

## TODO

 - [ ] Considering adding DTLN noise cancellation tflite model to improve whisper ASR accuracy in noisy environments.

## Here are links for comparing TFLite with other Whisper models:

[Whisper's Comparative Analysis](https://alphacephei.com/nsh/2022/12/11/whisper-other.html)
[Speech Recognition Experiments Repository](https://github.com/fquirin/speech-recognition-experiments)
[OpenVoiceOS' Whisper TFLite Plugin](https://github.com/OpenVoiceOS/ovos-stt-plugin-whisper-tflite)

## Stay Updated

Stay connected to this repository for further developments and updates related to the Whisper enhanced TFLite model. We are constantly working to improve its performance and compatibility with various edge devices.

If you have any questions or encounter any issues, please don't hesitate to open an issue in this repository. We'll be happy to assist you!

## Citing:

If you are using the DTLN model, please cite:

```bash
@inproceedings{Westhausen2020,
  author={Nils L. Westhausen and Bernd T. Meyer},
  title={{Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={2477--2481},
  doi={10.21437/Interspeech.2020-2631},
  url={http://dx.doi.org/10.21437/Interspeech.2020-2631}
}
```
