# Packages to import :

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


# Data set-up :

main_dir = "top_view_vials_002"
classes=["airnob", "csp","nothing"]
batch_number=15
train_transform = transforms.Compose([transforms.Resize( (224,224) ),
                                      transforms.RandomRotation(30),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])  ])
test_transform = transforms.Compose([transforms.Resize( (224,224) ),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])  ])

train_data = datasets.ImageFolder(main_dir, transform= train_transform)
test_data =datasets.ImageFolder(main_dir+"_test", transform= test_transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_number, shuffle=True)                                                         
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_number)  

# Functions:

def image_visualize(image, label):
    global classes
    if torch.is_tensor(image):
        image = image.numpy().transpose((1,2,0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array ([0.229, 0.224, 0.225])
        image = std*image+mean
        plt.title(classes[label])
        plt.imshow(image)
        plt.show()


# Visualize data:

# images, labels = next(iter(train_loader))
# # print(images[0].shape)
# image_visualize(images[0], labels[0])

# Model creation:
model = models.densenet121(pretrained=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for param in model.parameters():
    param.requires_grad= False 

model.classifier = nn.Sequential( nn.Linear(1024,256),
                                  nn.ReLU(),
                                  nn.Dropout(0.4),
                                  nn.Linear(256,3) ,
                                  nn.LogSoftmax(dim=1) )

model=model.to(device); # Optimizer goes after always.
criterion = nn.NLLLoss()
optimizer =optim.Adam(model.classifier.parameters(), lr=0.003)


epochs = 20

steps = 0

running_loss = 0

print_every = 5

for epoch in range(epochs):

    for inputs, labels in train_loader:

        steps += 1

        # Move input and label tensors to the default device

        inputs, labels = inputs.to(device), labels.to(device)

       

        optimizer.zero_grad()

       

        logps = model.forward(inputs)

        loss = criterion(logps, labels)

        loss.backward()

        optimizer.step()

 

        running_loss += loss.item()

       

        if steps % print_every == 0:

            test_loss = 0

            accuracy = 0

            model.eval()

            with torch.no_grad():

                for inputs, labels in test_loader:

                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)

                    batch_loss = criterion(logps, labels)

                   

                    test_loss += batch_loss.item()

                   

                    # Calculate accuracy

                    ps = torch.exp(logps)

                    top_p, top_class = ps.topk(1, dim=1)

                    equals = top_class == labels.view(*top_class.shape)

                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    

            print(f"Epoch {epoch+1}/{epochs}.. "

                  f"Train loss: {running_loss/print_every:.3f}.. "

                  f"Test loss: {test_loss/len(test_loader):.3f}.. "

                  f"Test accuracy: {accuracy/len(test_loader):.3f}")

            running_loss = 0

            model.train()


save_path= "/home/alvaro/Desktop/pyPro/opencv_program"
torch.save(model.state_dict(), save_path+"/"+"Vial_CNN_test.pth")

########## Getting Data Straight : #############

# Train Images  == 560

# Batch Size == 32 , shuffle == True

# Number of batches == 17.5