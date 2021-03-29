import os
import cv2


def nextidx(subdirs):
    subdirs_index = []
    if len(subdirs)>0:
        for i in subdirs:
            for x,j in enumerate(i):
        
             if j.isdigit():
                index0=i[x:]
                subdirs_index.append(index0) 
                break
        for u,k in enumerate(subdirs_index):
            index1=k.find(".")
            subdirs_index[u]=int(k[0:index1])
    else:
        subdirs_index=[0]
    return max(subdirs_index)+1


def directory_check_create(path):
    if not os.path.exists(path):
        print("Folder(s) had to be created")
        os.makedirs(path)
    else:
            print("Folder already exists")



# test_path=os.listdir("/home/alvaro/Desktop/pyPro/opencv_program/top_view_vials/csp")

# max([0])
### Bars Plot function

def result_output(outputs_data):

    ps_0=outputs_data.numpy().squeeze()

   

    return ps_0

 

 

### Images Plot function

 

def result_image(image):

    # image represents a single image

    # image needs to be a tensor

    if torch.is_tensor(image) and image.shape[0]>0 and image.shape[0]<4:

       

        im=image.numpy().transpose((1,2,0))

        mean = np.array([0.485, 0.456, 0.406])

        std = np.array([0.229, 0.224, 0.225])

        im = std * im + mean

        im = np.clip(im, 0, 1)

        print("image is ready to be plotted by plt library")

    else:

        print("This image is not a Tensor")

    return im

def cv2_image_to_plt(path):
    # string needs to have file extension

    if isinstance(path,str):

         
        image=cv2.imread(path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    else:
        print("The argument is not a string type variable")

    return image

# cv2_image_to_plt("nvidia_work/csp.jpg")
