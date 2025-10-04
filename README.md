# 🕵🏻‍♂️ AI CCTV 📹
This cross-platform, offline app uses `OnnxRuntime` (Machine Learning) to detect any `HUMAN` in the `Camera` frame in a loop and saves the file in app specific path if it detects, else it doesn't save the frame. It can also send a `SMS` alert to a given number if you are on `Android`.

> This project is buid on `kivy`, `kivymd` and uses `onnxruntime`, `numpy`, `opencv` etc. to perform the tasks. Our approach is always `Offline first` and `Local AI` (you only need internet for the first time to downlaod the tiny `onnx` model, app will prompt you). This project may have bugs & please let us know if you face any.

### Features
1. Detects if there is any human in a frame for a second (checks five frames in a second). Uses Artificial Intelligence Locally without exposing anything to the ineternet.

2. Sends sms alert to given number if detection is successful and depends on the frequency (minimum frequency is 1min, so that it doen't send too many messages, default is 10min). Available only on `Android`.
    > But, it will save the frames irrespective of sms frequency.

3. `No Ads`, completely `open-source` (free).
4. No `trackers`, no `data collection` etc. Actually you own your data & your app.

## 📽️ Demo
You can click on the below Image or this [Youtube Link](https://www.youtube.com/watch?v=Sbc2sClECdk) to see the demo. Please let me know in the comments, how do you feel about this App. <br>
[![AI-CCTV](./docs/images/thumb.png)](https://www.youtube.com/watch?v=Sbc2sClECdk)

## 🖧 Our Scematic Architecture
To be added...

## 🧑‍💻 Quickstart Guide

### 📱 Download & Run the Android App
You can check the [Releases](https://github.com/daslearning-org/ai-surveillance/tags) and downlaod the latest version of the android app on your phone.

#### Requirements
1. Minimum android verion: `9`

### 💻 Download & Run the Windows or Linux App
To be built later.

### 🐍 Run with Python

1. Clone the repo
```bash
git clone https://github.com/daslearning-org/ai-surveillance.git
```

2. Run the application
```bash
cd ai-surveillance/app/
pip install -r requirements.txt # virtual environment is recommended
python main.py
```

## 🦾 Build your own App
The Kivy project has a great tool named [Buildozer](https://buildozer.readthedocs.io/en/latest/) which can make mobile apps for `Android` & `iOS`

### 📱 Build Android App
A Linux environment is recommended for the app development. If you are on Windows, you may use `WSL` or any `Virtual Machine`. As of now the `buildozer` tool works on Python version `3.11` at maximum. I am going to use Python `3.11`

```bash
# add the python repository
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# install all dependencies.
sudo apt install -y ant autoconf automake ccache cmake g++ gcc libbz2-dev libffi-dev libltdl-dev libtool libssl-dev lbzip2 make ninja-build openjdk-17-jdk patch patchelf pkg-config protobuf-compiler python3.11 python3.11-venv python3.11-dev

# optionally we can default to python 3.11
sudo ln -sf /usr/bin/python3.11 /usr/bin/python3
sudo ln -sf /usr/bin/python3.11 /usr/bin/python
sudo ln -sf /usr/bin/python3.11-config /usr/bin/python3-config

# optionally you may check the java installation with below commands
java -version
javac -version

# install python modules
git clone https://github.com/daslearning-org/ai-surveillance.git
cd ai-surveillance/app/
python3.11 -m venv .env # create python virtual environment
source .env/bin/activate
pip install -r req_android.txt

# build the android apk
buildozer android debug # this may take a good amount of time for the first time & will generate the apk in the bin directory
```
