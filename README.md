# FaceCTRL: control your media player with your face

After being interrupted dozens of times a day while I was coding and listening to my favorite music, I decided to find a solution that eliminates the stress of stopping and re-starting the music.

The idea is trivial:

- When you're in front of your PC with your headphones on: the music plays.
- Someone interrupts you, and you have to remove your headphones: the music pause.
- You walk away from your PC: the music pause.
- You come back to your PC, and you put the headphones on: the music plays again.

However, the manual control of your player is still possible. If you decide to pause the music while you're still in front of your PC with your headphones on, the control of the media player is transferred to the player itself. To give back the control to playerctrl, just walk away from your PC (or hide the image captured from your webcam using a finger, for some second).

FaceCTRL takes control of your webcam as soon as you open your media player, and releases it when you close the player.

## Requirements

- A webcam
- [Playerctl](https://github.com/altdesktop/playerctl) installed (`pacman -S playerctl` on Archlinux)
- Python >= 3.7
- OpenCV is not required to be installed system-wise, but it is recommended. The python package of OpenCV doesn't contain the pre-trained models for face localization (XML files) and you have to download them from the [OpenCV repository](https://github.com/opencv/opencv/). OpenCV installed system-wise, instead, usually ships them in the `/usr/share/opencv4/haarcascades/` folder.

### Installation

If you just want to use this tool without making any change, you use pip:

```
pip install --upgrade facectrl
```

Please note that this software is still alpha software.

For **development**: clone the repository and just `pip install -e .`

## Usage

The project does not ship a pre-trained model; you have to train a model by yourself and use it.

Why? Because I don't have enough data of people with and without headphones to train a model able to generalize well. If you're interested in contributing by sharing your dataset (to remove the training phase and give to the user a ready to use model), please open an issue.

### Dataset creation

The process of dataset creation is entirely automatic. Just execute:

```
python -m facectrl.dataset \
           --dataset-path ~/face \
           --classifier-params /usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml
```

where:

- `--dataset-path` is the destination folder of your dataset. It contains 2 folders (`on` and `off`) with the captured images with headphones on and off.
- `--classifier-params` is the path of the XML file containing your face detector parameters (Viola-Jones Haar classifier)

Follow the instructions displayed in the terminal.

**Hint**: move in front of the camera until you see your face in a window with an increasing number on the bottom right corner. Your face is now being tracked, thus try to acquire as many images as possible with different appearances. Acquire at least 1000 images with headphones on and 1000 images with headphones off.


If you want to share your dataset, please, open an issue! In this way, we can reach the goal of shipping a well-trained model together with FaceCTRL.

### Training

You can train the 3 models available with this simple bash script:

```
for model in ae vae classifier; do
    python -m facectrl.ml.train --dataset-path ~/face/ --logdir ~/log_$model --epochs 100  --model $model
done
```

where:

- `--dataset-path` is the path of your training dataset (see [Dataset creation](#dataset-creation)).
- `--logdir` is the path of your trained model. This folder contains the logs (use tensorboard to see the training progress/result `tensorboard --logdir $path`), and the model that reached the highest validation performance converted in SavedModel file format.


### Execution

The execution is straightforward, and I highly recommend to put this script in the startup script of your system (it's easy with systemd).

**NOTE**: you must execute this script **before** starting your media player.

```
python -m facectrl.ctrl \
           --logdir ~/log/ \
           --player spotify \
           --classifier-params /usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml \
           --metric binary_accuracy
```

where:

- `--logdir` is the log directory specified during the training.
- `--player` is the media player to control.
- `--classifier-params` is the path of the XML file containing your face detector parameters (Viola-Jones Haar classifier). Use the same parameters using during the [Dataset creation](#dataset-creation).
- `--metric` is the metric used during the training for the model selection. For the classifier model is the `binary_accuracy`, for the `vae` and `ae` model is the `AEAccuracy`.
