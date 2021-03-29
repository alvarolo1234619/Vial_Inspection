##################### Import packages #####################

import cv2
import numpy as np
import jetson.inference
import jetson.utils
import torch
import torchvision
from torch import optim
from torch import nn
from torchvision import transforms, models
from collections import OrderedDict
from  utils_dlinano import preprocess


##################### Variables #####################



dispW=640
dispH=480
flip=2


# Window name in which image is displayed 
window_name = 'Image'  
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (50, 50) 
# fontScale 
fontScale = 1
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2
# Classes below
classes=["airnob", "csp","nothing"]


##################### Loading the Model ##################

model=models.densenet121()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.classifier = nn.Sequential( nn.Linear(1024,256),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(256,3) ,
                                  nn.LogSoftmax(dim=1) )

for param in model.parameters():
    param.requires_grad= False 


CNN_model = model
CNN_model.load_state_dict (torch.load("Vial_CNN.pth"))
CNN_model.to(device)
CNN_model.eval()


##################### Setting the Camera ##################
camSet='nvarguscamerasrc wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.5 brightness=-.2 saturation=1.2 ! appsink'

cam = cv2.VideoCapture(camSet)





########################

while True:
    ret, frame =cam.read()
    
    
    img = preprocess(frame)
    out=CNN_model(img)
    _,preds_tensor = torch.max(out,1)
    preds = np.squeeze(preds_tensor.to("cpu").numpy())
    prediction_text =classes[preds]


    frame = cv2.putText(frame, prediction_text, org, font,  
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("Vial Recognition",frame)

    if cv2.waitKey(1)== ord("q"):
        break

cam.release()
cv2.destroyAllWindows()