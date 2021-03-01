# Vial_Inspection
This repository contains all the necessary code for a Vial Inspection Application.The code is presented in a Jupyter notebook format. 

## Background
The application/inspection consists in inspecting a vial so that a later process can use this vial. If a vial is determined to be failed by this model, then the vial would be rejected to the fail bin. The Pass/Fail criteria is summarized below:

### Pass
  * Vial lid is pointing/facing to the left in the Field Of View (FOV); **and**
  * Vial is closed. See images below for a pictorial representation

### Fail
  * Vial lid is pointing in the wrong direction (right in the FOV); **or/and**
  * Vial is open. See images below for a pictorial representation

## Repository Contents
The contents of this repository are :
  * Train Folder : this folder contains two (2) subdirectories: pass, and fail. 
  * Test Folder : this folder contains two (2) subdirectories (pass and fail) with image data for you to forward pass (test) the model
  * My_Vial_CNN.ipynb : this Jupyter Notebook contains the code where the model is trained. Here the bias and weight parameters of the model get updated through the epochs
  * VialCNN.pth : This is the already trained model
  * Applying VialCNN.ipynb : This is where the model is deployed
