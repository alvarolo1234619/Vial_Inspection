##################### Import packages #####################
import torch
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from torch import nn
from utils_dlinano import tensor2plt
from utils_dlinano import preprocess
import numpy as np

root = tk.Tk()
root.withdraw()
##################### Preprocess Data #####################
test_transform = transforms.Compose([transforms.Resize( (224,224) ),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])  ])
batch_number = 5

# List contains images that will be plotted:
plot_images_list=list(0. for i in range (batch_number))
predictions_list = list(0. for i in range (batch_number))
scores_list= list(0. for i in range(batch_number))

# Data Loading
directory = filedialog.askdirectory()
test_data = datasets.ImageFolder(directory, transform=test_transform) # No transforms, you do need transforms
testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_number, shuffle= True)

images, labels = next(iter(testloader))






##################### Loading the Model ##################

model=models.densenet121()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.classifier = nn.Sequential( nn.Linear(1024,256),
                                  nn.ReLU(),
                                  nn.Dropout(0.4),
                                  nn.Linear(256,3) ,
                                  nn.LogSoftmax(dim=1) )

for param in model.parameters():
    param.requires_grad= False


CNN_model = model
CNN_model.load_state_dict (torch.load("Vial_CNN.pth"))
CNN_model.to(device)
CNN_model.eval()


##################### Model Forward ##################


for index, img in enumerate(images):

    # The image being passed to preprocess is a Tensor
    # img=preprocess(image)
    image=img.detach().clone()
    image = image[None, :, :, :] # Adds a dimension
    image=image.to("cuda")
    out = CNN_model(image)

    ps = torch.exp(out)
    top_p, top_class = ps.topk(1, dim=1)

    score_tensor, preds_tensor = torch.max(out, 1)
    preds = np.squeeze(preds_tensor.to("cpu").numpy())
    prediction_text = test_data.classes[preds]

    predictions_list[index] = prediction_text
    scores_list[index]=ps.data.to("cpu").numpy()

    # Images to be plotted below
    image_plt=tensor2plt(img)
    plot_images_list[index]= image_plt


########################

##################### Generating Plots ##################
fig, ([[ax1, ax2],[ax3, ax4], [ax5, ax6], [ax7,ax8], [ax9,ax10]]) = plt.subplots(figsize=(6,9) , ncols = 2, nrows=5)
ax1.imshow(plot_images_list[0])
ax1.axis("off")

ax2.barh(np.arange(3), scores_list[0].flatten() )
ax2.set_aspect(0.1)
ax2.set_yticks(np.arange(3))
ax2.set_yticklabels(test_data.classes)

ax3.imshow(plot_images_list[1])
ax3.axis("off")

ax4.barh(np.arange(3), scores_list[1].flatten() )
ax4.set_aspect(0.1)
ax4.set_yticks(np.arange(3))
ax4.set_yticklabels(test_data.classes)

ax5.imshow(plot_images_list[2])
ax5.axis("off")

ax6.barh(np.arange(3), scores_list[2].flatten() )
ax6.set_aspect(0.1)
ax6.set_yticks(np.arange(3))
ax6.set_yticklabels(test_data.classes)


ax7.imshow(plot_images_list[3])
ax7.axis("off")

ax8.barh(np.arange(3), scores_list[3].flatten() )
ax8.set_aspect(0.1)
ax8.set_yticks(np.arange(3))
ax8.set_yticklabels(test_data.classes)

ax9.imshow(plot_images_list[4])
ax9.axis("off")

ax10.barh(np.arange(3), scores_list[4].flatten() )
ax10.set_aspect(0.1)
ax10.set_yticks(np.arange(3))
ax10.set_yticklabels(test_data.classes)

plt.tight_layout()
plt.show(block=False)
plt.pause(20)
plt.close()

