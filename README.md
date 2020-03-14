# FaceCTRL: control your media player with your face.
---


## Usage


### Training


### Execution

You can invoke `facectrl.ctrl` specifying:

- `logdir` is the log directory specified during the training
- `player` is the media player to control
- `classifier-params` is the path of the XML file containing your face detector parameters (Viola-Jones Haar classifier)
- `metric` is the metric used during the training for the model selection

Please **note**: you must execute this script **before** starting your media player.

**Example**: that's what I do when I want to use the classifier model trained to solve a binary classification problem and the Haar cascade classifier (frontalface alt 2).

```
python -m facectrl.ctrl \
           --logdir ~/log/ \
           --player spotify \
           --classifier-params /usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml \
           --metric binary_accuracy
```
