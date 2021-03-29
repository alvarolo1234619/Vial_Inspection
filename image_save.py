##################### Import packages #####################

import cv2
import os
from DirectoryFunctions import nextidx
from DirectoryFunctions import directory_check_create

##################### Variables #####################

# main_path0 is the path to the folder
# to be used for saving images to test and train your model
# change the string characters before the / to save vials wherever you would
# like
main_path0="HelloWorld/"
directory_check_create(main_path0)

saving_folder="nothing"
directory_check_create(main_path0+saving_folder+"/")

image_filename = saving_folder
filename_extension=".jpg"



my_sub_directories =os.listdir() 

# The following variables width, height, etc. exist to set up your camera
# I am currently using a Raspberry Pi Camera Module V2
# Currently my camera resolution is width=3264, height=2464
# Display resolution in monitor is set by dispW and dispH
dispW=1280
dispH=720
flip=2
camSet='nvarguscamerasrc wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.5 brightness=-.2 saturation=1.2 ! appsink'
pi_cam=cv2.VideoCapture(camSet)


##################### Functions #####################
# click_event() function will allow you to save camera captures at your designated directory.
# Your designated directory is given by main_path0 and saving_folder variables.
# The images saved are in .jpg file format
def click_event(event, x, y, flags, param):
    # global index
    if event == cv2.EVENT_LBUTTONDOWN:
        index=nextidx(os.listdir(main_path0+saving_folder))
        cv2.imwrite(main_path0 + saving_folder + "/"+image_filename+str(index)+filename_extension, frame)
        

##################### Program #####################
##################### Functionality #####################
# To exit this while loop press the key "q" for quitting/breaking the loop
# WARNING do not exit the loop by pressing the x or Alt+F4. This will not break the loop
# To acquire images at the directory you set earlier, click on the window "piCam Display"

while True:
    ret, frame = pi_cam.read()

    if cv2.waitKey(1)== ord("q"):
        break
    elif cv2.waitKey(1)==ord("c"):
        index=nextidx(os.listdir(main_path0+saving_folder))
        cv2.imwrite(main_path0 + saving_folder + "/"+image_filename+str(index)+filename_extension, frame)

    cv2.imshow("piCam Display", frame)
    cv2.setMouseCallback("piCam Display", click_event)
   
        


##################### End of Program #####################
# These lines of code (below) release the resources
pi_cam.release()
cv2.destroyAllWindows()