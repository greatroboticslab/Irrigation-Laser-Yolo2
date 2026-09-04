import torch
from ultralytics import YOLO

from roboflow import Roboflow
rf = Roboflow(api_key="tnmtibY8WgTiWt4wjddx")
project = rf.workspace("mtsu").project("irrigation-flow-rate")
dataset = project.version(1).download("yolov5")


# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', autoshape=False, pretrained=False)  # load scratch

# Images
imgs = ['Data/train/images/Vid1_Set_mp4-1_jpg.rf.39b20b524476df214538cc54614fdbc3.jpg']  # batch of images

# Inference
results = model(imgs)



# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
#results.pandas().xyxy[0]  # img1 predictions (pandas)