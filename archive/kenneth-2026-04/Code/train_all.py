import os
import subprocess
import sys

files = os.listdir("/projects/kp9e/Irrigation-Laser-Yolo2/Data/training-data/")

for name in files:

    subfiles = os.listdir("/projects/kp9e/Irrigation-Laser-Yolo2/Data/training-data/" + name + "/")

    for subfile in subfiles:

        print("projects/kp9e/Irrigation-Laser-Yolo2/Data/training-data/" + name + "/" + subfile + "/")
        result = subprocess.run(
            [sys.executable,
             "train.py",
             "--img",
             "640",
             "--epochs",
             "3",
             "--data",
             "projects/kp9e/Irrigation-Laser-Yolo2/Data/training-data/" + name + "/" + subfile + "/" + "data.yaml",
             "--weights",
             "yolov5s.pt",
                          "--project",
                          "runs/train/" + name + "/",
                          "--name",
                          subfile]
                     )