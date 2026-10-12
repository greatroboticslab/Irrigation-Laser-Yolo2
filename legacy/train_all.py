import os
import subprocess

files = os.listdir("data/training-data/")

for name in files:

    subfiles = os.listdir("data/training-data/" + name + "/")

    for subfile in subfiles:

        print("data/training-data/" + name + "/" + subfile + "/")
        result = subprocess.run(
            ["python",
             "train.py",
             "--img",
             "640",
             "--epochs",
             "3000",
             "--data",
             "data/training-data/" + name + "/" + subfile + "/" + "data.yaml",
             "--weights",
             "yolov5s.pt",
             "--project",
             "runs/train/" + name + "/",
             "--name",
             subfile],
            capture_output=True,
            text=True
        )