import csv
import json
from pprint import pprint
import collections
import scipy.io
import glob
import cv2
import ast
# from PIL import Image
import os
import numpy as np
from torch.utils.data import DataLoader
from datasetIMG import DataLoaderInstanceSegmentation
from pathlib import Path

def data_grouping_GWHD():
    result = {}
    with open('train.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # This skips the first row of the CSV file.
        next(csvreader)
        for row in csvreader:
            if row[0] in result:
                result[row[0]] = result[row[0]] + 1
            else:
                result[row[0]] = 1

    # print(result)
    result_count = {}
    for key, value in result.items():
        if value in result_count:
            result_count[value] = result_count[value] + 1
        else:
            result_count[value] = 1
    result_count = collections.OrderedDict(sorted(result_count.items()))
    with open('gwhd_distribution.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        i = 1
        for key, value in result_count.items():
            writer.writerow([key, value])
            i = i + 1

def data_grouping_COCO():
    with open('panoptic_val2017.json') as f:
        data = json.load(f)
    pprint(len(data['annotations'][5]['segments_info']))

    result_count = {}
    for data in data['annotations']:
        if len(data['segments_info']) in result_count:
            result_count[len(data['segments_info'])] = result_count[len(data['segments_info'])] + 1
        else:
            result_count[len(data['segments_info'])] = 1
    result_count = collections.OrderedDict(sorted(result_count.items()))
    with open('coco_distribution.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        i = 1
        for key, value in result_count.items():
            writer.writerow([key, value])
            i = i + 1

def data_grouping_BSR():
    result_count = {}
    matsloc = glob.glob("BSRtrain200_fromBSD500/*.mat")
    for matloc in matsloc:
        mat = scipy.io.loadmat(matloc)
        # print(len(mat['groundTruth'][0]))
        if len(mat['groundTruth'][0]) in result_count:
            result_count[len(mat['groundTruth'][0])] = result_count[len(mat['groundTruth'][0])] + 1
        else:
            result_count[len(mat['groundTruth'][0])] = 1

    result_count = collections.OrderedDict(sorted(result_count.items()))
    with open('BSR_distribution.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        i = 1
        for key, value in result_count.items():
            writer.writerow([key, value])
            i = i + 1        



def show_image_withAnnotation():    

    with open('train.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # This skips the first row of the CSV file.
        next(csvreader)
        filename, file_extension = os.path.splitext('6fb67942f.jpg')
        img = cv2.imread(filename+file_extension)
        for row in csvreader:
            if (row[0]== filename):
                pts = ast.literal_eval(row[3])
                x1 = int(pts[0])
                y1 = int(pts[1])
                x2 = int(pts[0] + pts[2])
                y2 = int(pts[1] +pts[3])
                color = list(np.random.random(size=3) * 256)
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 4)
        cv2.imwrite(filename+"_ann"+file_extension,img)
        # cv2.imshow("lalala", img)
        k = cv2.waitKey(0) # 0==wait forever


import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib inline
# For adding noise to images
noise_factor=0.5
batch_size = 5 

# Define the NN architecture
class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 1 --> 32), 3x3 kernels
        # self.conv1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  
        # conv layer (depth from 32 --> 16), 3x3 kernels
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        # conv layer (depth from 16 --> 8), 3x3 kernels
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(2, 2)
        
        ## decoder layers ##
        # transpose layer, a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        # self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)  # kernel_size=3 to get to a 7x7 image output
        self.t_conv1 = nn.ConvTranspose2d(8, 8, 2, stride=2)  # kernel_size=3 to get to a 7x7 image output
        # two more transpose layers with a kernel of 2
        # self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        # self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 3, stride=2)
        # one, final, normal conv layer to decrease the depth
        # self.conv_out = nn.Conv2d(32, 1, 3, padding=1)
        self.conv_out = nn.Conv2d(32, 3, 3, padding=1)
        
    def forward(self, x):
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # add third hidden layer
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # compressed representation
        
        ## decode ##
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        # transpose again, output should have a sigmoid applied
        x = F.sigmoid(self.conv_out(x))
                
        return x# initialize the NN



# the autoencoder network
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # # encoder layers
        # self.enc1 = nn.Conv2d(3, 512, kernel_size=3, padding=1)
        # self.enc2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # self.enc3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # self.enc4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # # decoder layers
        # self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  
        # self.dec2 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
        # self.dec3 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
        # self.dec4 = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        # self.out = nn.Conv2d(512, 3, kernel_size=3, padding=1)
        
        # self.bn1 = nn.BatchNorm2d(512)
        # self.bn2 = nn.BatchNorm2d(256)
        # self.bn3 = nn.BatchNorm2d(128)
        # self.bn4 = nn.BatchNorm2d(64)        
        # self.pool = nn.MaxPool2d(2, 2)

        # encoder layers
        self.enc0 = nn.Conv2d(3, 512, kernel_size=3, padding=1)
        self.enc1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # self.enc3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # self.enc4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        # self.enc5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        
        # decoder layers
        # self.dec0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # self.dec1 = nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)  
        # self.dec2 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
        self.dec3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec4 = nn.ConvTranspose2d(128, 256, kernel_size=2, stride=2)
        self.dec5 = nn.ConvTranspose2d(256, 512, kernel_size=2, stride=2)
        self.out = nn.Conv2d(512, 3, kernel_size=3, padding=1)
        
        self.bn0 = nn.BatchNorm2d(512)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        # self.bn3 = nn.BatchNorm2d(256)
        # self.bn4 = nn.BatchNorm2d(128)
        # self.bn5 = nn.BatchNorm2d(64)        
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # encode
        x = F.relu(self.enc0(x))
        x = (self.bn0(x))
        x = self.pool(x)        
        x = F.relu(self.enc1(x))
        x = (self.bn1(x))
        x = self.pool(x)
        x = F.relu(self.enc2(x))
        x = (self.bn2(x))
        x = self.pool(x)
        # x = F.relu(self.enc3(x))
        # x = (self.bn3(x))
        # x = self.pool(x)
        # x = F.relu(self.enc4(x))
        # x = (self.bn4(x))
        # x = self.pool(x)
        # x = F.relu(self.enc5(x))
        # x = (self.bn5(x))
        # x = self.pool(x)
        # the latent space representation
        
        # decode
        # x = F.relu(self.dec0(x))
        # x = (self.bn5(x))
        # x = F.relu(self.dec1(x))
        # x = (self.bn4(x))
        # x = F.relu(self.dec2(x))
        # x = (self.bn3(x))
        x = F.relu(self.dec3(x))
        x = (self.bn2(x))
        x = F.relu(self.dec4(x))
        x = (self.bn1(x))
        x = F.relu(self.dec5(x))
        x = (self.bn0(x))        
        x = torch.sigmoid(self.out(x))
        return x


def data_loading():
    # convert data to torch.FloatTensor
    # transform = transforms.ToTensor()
    transform = transforms.Compose([transforms.Resize([int(1024), int(1024)]),
                                transforms.ToTensor()])
    # load the training and test datasets
    # train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    # train_data = datasets.ImageFolder( root="dataset", transform=transform)
    # # test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    # test_data = datasets.ImageFolder( root="dataset", transform=transform)
    # # Create training and test dataloaders
    num_workers = 0
    # # how many samples per batch to load
    
    # # prepare data loaders
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    # test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers) 

    train_dataset = DataLoaderInstanceSegmentation()
    test_dataset = DataLoaderInstanceSegmentation(train = False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)                            
        
    return train_loader, test_loader

def data_visualize(train_loader):
    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()
    
    # get one image from the batch
    img = np.squeeze(images[0])
    

    fig = plt.figure(figsize = (5,5)) 
    ax = fig.add_subplot(111)
    # ax.imshow(img, cmap='gray')
    img = np.transpose( img, (1, 2, 0))
    ax.imshow(img)
    # plt.axis('off')
    plt.savefig("Sample.jpeg", dpi=100)
    plt.show()

def training_model(model, train_loader):
    # Specify loss function
    criterion = nn.MSELoss()
    # sSecify loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # Number of epochs to train the model
    n_epochs = 5



    for epoch in range(1, n_epochs+1):
        # Monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data, noise_less in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            # images, _ = data
            images = data
            
            ## add random noise to the input images
            # noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            # noisy_imgs = images.clone()
            noisy_imgs = noise_less

            # for idx, image in enumerate(noisy_imgs):
            #     # image[image <= 0.39] = 0
            #     image = segmentation_with_masking((255*image).permute(1, 2, 0).numpy())  
            #     noisy_imgs[idx] = torch.from_numpy(image).permute(2, 0, 1)
              
            

            # Clip the images to be between 0 and 1
            # noisy_imgs = np.clip(noisy_imgs, 0., 1.)
                    
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            ## forward pass: compute predicted outputs by passing *noisy* images to the model
            outputs = model(images)
            # outputs = model(noisy_imgs[None, ...])
            # calculate the loss
            # the "target" is still the original, not-noisy images
            loss = criterion(outputs, noisy_imgs)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, 
            train_loss
            ))    

def testing_model(model, test_loader):
    # obtain one batch of test images
    # dataiter = iter(test_loader)
    dataiter = iter(test_loader)
    # images, labels = dataiter.next()
    images = dataiter.next()
    names = images[1]
    # add noise to the test images
    # noisy_imgs = images + noise_factor * torch.randn(*images.shape)
    # noisy_imgs = images
    noisy_imgs = images[0].clone()
    # for idx, image in enumerate(noisy_imgs):
    #     image[image <= 0.39] = 0
    #     noisy_imgs[idx] = image  

    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    # get sample outputs
    output = model(noisy_imgs)
    # prep images for display
    noisy_imgs = noisy_imgs.numpy()
    # output is resized into a batch of iages
    output = output.view(batch_size, 3, 1024, 1024)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()
    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))
    # input images on top row, reconstructions on bottom
    im_no = 0
    for noisy_imgs, row in zip([noisy_imgs, output], axes):
        for img, ax in zip(noisy_imgs, row):
            img = np.transpose( img, (1, 2, 0))
            im = Image.fromarray(((np.squeeze(img))* 255).astype(np.uint8))
            if im.mode != 'RGB':
                im = im.convert('RGB')
                
            # # Separate RGB arrays
            # R, G, B = im.convert('RGB').split()
            # r = R.load()
            # g = G.load()
            # b = B.load()
            # w, h = im.size
            # # Convert non-white pixels to black
            # for i in range(w):
            #     for j in range(h):
            #         if(r[i, j] < 100 or g[i, j] < 100 or b[i, j] < 100):
            #             r[i, j] = 0 # Just change R channel
            # # Merge just the R channel as all channels
            # im = Image.merge('RGB', (R, R, R)) 

            im.save('outfile'+str(im_no)+'.png')
            im_no = im_no + 1
            # ax.imshow(np.squeeze(img), cmap='gray')
            ax.imshow(img)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
    plt.show()




def background_less_images(model, test_loader):
    folder_path="images_testing"
    for data, labels in test_loader:    
        images = data
        names = labels
        noisy_imgs = images.clone()
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        # get sample outputs
        output = model(noisy_imgs)
        # prep images for display
        noisy_imgs = noisy_imgs.numpy()
        # output is resized into a batch of iages
        output = output.view(len(names), 3, 1024, 1024)
        # use detach when it's an output that requires_grad
        output = output.detach().numpy()
        for img, name in zip(output, names):
            img = np.transpose( img, (1, 2, 0))
            im = Image.fromarray(((np.squeeze(img))* 255).astype(np.uint8))
            if im.mode != 'RGB':
                im = im.convert('RGB')
            target_loc = os.path.join(folder_path,'backless_images',name)    
            im.save(target_loc)

def denoising_autoencoder():
    # Load Data
    train_loader, test_loader = data_loading()        
    # Visualize Data
    data_visualize(train_loader)
    # Construct Model
    # model = ConvDenoiser()
    model = ConvAutoencoder()
    
    print(model)
    # Training
    training_model(model, train_loader)
    # model_dir = Path('model')
    # modelname = 'model.pth'
    # torch.save(model.state_dict(), model_dir.joinpath(modelname))
    
    # Testing
    
    # model.eval()
    # model_dir = Path('model')
    # model_path = model_dir.joinpath('model.pth')
    # param = torch.load(model_path)
    # model.load_state_dict(param)
    # testing_model(model, test_loader)
    background_less_images(model, test_loader)


import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
from tqdm.notebook import tqdm
np.random.seed(1)
import os

def data_loading_for_masking():
    paths = glob.glob('dataset/Class1/*.jpg',recursive=True)
    print(len(paths))
    orig = np.array([np.asarray(Image.open(img)) for img in tqdm(paths)])
    orig.shape
    return orig, paths

def vizualize_images(images, name="Original"):
    plt.figure(figsize=(9,9))
    for i, img in enumerate(images[0:16]):
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        # plt.imshow(threshimg,cmap='gray')
        # plt.imshow(cv2.cvtColor(edge, cv2.COLOR_GRAY2RGB))
        # plt.imshow(maskimg, cmap='gray')
        # plt.imshow(cv2.cvtColor(segimg, cv2.COLOR_BGR2RGB))   
    plt.suptitle(name, fontsize=20)
    plt.show()


def write_segmented_images(segmented, paths):
    for i, image in tqdm(enumerate(segmented)):
        directory = paths[i].rsplit('/', 3)[0] + '/segmented/' + paths[i].rsplit('/', 2)[1]+ '/'
        os.makedirs(directory, exist_ok = True)
        cv2.imwrite(directory + paths[i].rsplit('/', 2)[2], image)

def segmentation_with_masking_all():
    orig, paths = data_loading_for_masking()
    vizualize_images(orig)
    gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in tqdm(orig)])
    vizualize_images(gray, "Grayscale")
    thresh = [cv2.threshold(img, np.mean(img), 255, cv2.THRESH_BINARY_INV)[1] for img in tqdm(gray)]
    np.mean(gray[0])
    vizualize_images(thresh, "Threshold")
    edges = [cv2.dilate(cv2.Canny(img, 0, 255), None) for img in tqdm(thresh)]
    vizualize_images(thresh, "Edges")

    masked=[]
    segmented=[]
    for i, img in tqdm(enumerate(edges)):
        cnt = sorted(cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
        # mask = np.zeros((256,256), np.uint8)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        masked.append(cv2.drawContours(mask, [cnt],-1, 255, -1))
        dst = cv2.bitwise_and(orig[i], orig[i], mask=mask)
        segmented.append(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

    vizualize_images(masked, "Mask")
    vizualize_images(segmented, "Segmented")
    write_segmented_images(segmented, paths)

def segmentation_with_masking(img):
    gray = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)[1]
    np.mean(gray)
    edges = cv2.dilate(cv2.Canny(thresh.astype(np.uint8), 0, 255), None)

    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    # mask = np.zeros((256,256), np.uint8)
    mask = np.zeros(edges.shape[:2], dtype=np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
    dst = cv2.bitwise_and(img, img, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return segmented

def mask_cropped(image, thresh_mask):
    result = cv2.bitwise_and(image, image, mask=thresh_mask)
    return result


# For cropped masking 
# folder_path="inrae_1_all"
# img_files = glob.glob(os.path.join(folder_path,"images","*.jpg"))
# for img_path in img_files:
#     img = np.asarray(Image.open(img_path).convert('RGB'))
#     mask = np.asarray(Image.open(os.path.join(folder_path,'masks',os.path.basename(img_path))).convert('L'))
#     cropped = mask_cropped(img, mask)
#     cv2.imwrite(os.path.join(folder_path,'croped_masks',os.path.basename(img_path)) , cropped)


# data_grouping_COCO()    
# data_grouping_GWHD()
# data_grouping_BSR()
# show_image_withAnnotation()
denoising_autoencoder()
# segmentation_with_masking_all()

from ochumanApi.ochuman import OCHuman
import cv2, os
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = (15, 15)
import ochumanApi.vis as vistool
from ochumanApi.ochuman import Poly2Mask
import cv2
# from google.colab.patches import cv2_imshow
from tqdm.notebook import tqdm
import numpy as np
ImgDir = 'images/'

def get_segmentation(data):
    img = cv2.imread(os.path.join(ImgDir, data['file_name']))
    height, width = data['height'], data['width']

    colors = [[255, 0, 0], 
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255], 
            [0, 0, 255], 
            [255, 0, 255]]


    for i, anno in enumerate(data['annotations']):
        bbox = anno['bbox']
        kpt = anno['keypoints']
        segm = anno['segms']
        max_iou = anno['max_iou']

        # img = vistool.draw_bbox(img, bbox, thickness=3, color=colors[i%len(colors)])
        if segm is not None:
            mask = Poly2Mask(segm)
            img = vistool.draw_mask(img, mask, thickness=3, color=colors[i%len(colors)])
        # if kpt is not None:
        #     img = vistool.draw_skeleton(img, kpt, connection=None, colors=colors[i%len(colors)], bbox=bbox)
    return img

def get_segmentation_GWHD(data):
    img = cv2.imread(os.path.join(ImgDir, data['file_name']))
    height, width = data['height'], data['width']

    colors = [[255, 0, 0], 
            [255, 255, 0],
            [0, 255, 0],
            [0, 255, 255], 
            [0, 0, 255], 
            [255, 0, 255]]


    for i, anno in enumerate(data['annotations']):
        bbox = anno['bbox']
        kpt = anno['keypoints']
        segm = anno['segms']
        max_iou = anno['max_iou']

        # img = vistool.draw_bbox(img, bbox, thickness=3, color=colors[i%len(colors)])
        if segm is not None:
            mask = Poly2Mask(segm)
            img = vistool.draw_mask(img, mask, thickness=3, color=colors[i%len(colors)])
        # if kpt is not None:
        #     img = vistool.draw_skeleton(img, kpt, connection=None, colors=colors[i%len(colors)], bbox=bbox)
    return img    

def mask_creation():    
    ochuman = OCHuman(AnnoFile='ochuman.json', Filter='segm')
    image_ids = ochuman.getImgIds()
    print ('Total images: %d'%len(image_ids))
    return ochuman, image_ids

def black_white_mask_creation(real_img, m_img):
    real_img = real_img.reshape(1, -1)[0]
    m_img = m_img.reshape(1, -1)[0]

    new = []

    for i, j in zip(real_img, m_img):
        if i != j:
            new.append(255) # human will white appear because of 255
        else:
            new.append(0) # background will black appear because of 0, set i instead of 0 to do not change backgraound

    new_np = np.array(new)
    new_np = new_np.reshape(512, 512, 3)

    return new_np    

# def color_mask_creation(real_img, m_img):
    blue_channel_r = real_img[:,:,0]
    green_channel_r = real_img[:,:,1]
    red_channel_r = real_img[:,:,2]
    
    blue_channel_m = m_img[:,:,0]
    green_channel_m = m_img[:,:,1]
    red_channel_m = m_img[:,:,2]
    
    new_b = []
    new_g = []
    new_r = []

    mks_img_new = np.zeros([512, 512, 3])
    
    for i in range(3):
        # print("i: ", i)
        if i == 0:
            img = blue_channel_r
            msk = blue_channel_m
        if i == 1:
            img = green_channel_r
            msk = green_channel_m
        if i == 2:
            img = red_channel_r
            msk = red_channel_m


        img = img.reshape(1, -1)[0]
        msk = msk.reshape(1, -1)[0]

        # print(f"Img: {img.shape}, Msk: {msk.shape}")

        if i == 0:
            new = new_b
        if i == 1:
            new = new_g
        if i == 2:
            new = new_r

        for k, j in zip(img, msk):
            if k != j:
                if i == 0:
                    new.append(50) # blue
                if i == 1:
                    new.append(235) # green
                if i == 2:
                    new.append(235) # red
            else:
                if i == 0:
                    new.append(120) # blue
                if i == 1:
                    new.append(0) # green
                if i == 2:
                    new.append(120) # red
    
    new_b = np.array(new_b).reshape(512, 512)
    new_g = np.array(new_g).reshape(512, 512)
    new_r = np.array(new_r).reshape(512, 512)

    mks_img_new[:,:,0] = new_b
    mks_img_new[:,:,1] = new_g
    mks_img_new[:,:,2] = new_r
    # print(mks_img_new.shape)      
    # print("Shape BGR: ", new_b.shape, new_g.shape, new_r.shape)  
    return mks_img_new    


def generator_images(batch_size, ind, ochuman):
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    while True:
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            data = ochuman.loadImgs(imgIds=['0000'+str(ind)])[0]
            file_name = data['file_name']
            img = cv2.imread(ImgDir+'/'+file_name)
            y = get_segmentation(data)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            y = cv2.resize(y, (IMG_WIDTH, IMG_HEIGHT))
            new = black_white_mask_creation(img, y)
            img = img / 255.
            y = new / 255.
            x_batch.append(img)
            y_batch.append(y)
        x_batch = np.array(x_batch)
        y_batch = {'seg': np.array(y_batch)
                    }
        yield x_batch, y_batch

def masking_all(ochuman):
    for i in [13,14,15,16,17,18,19,20,21,22]:
        for x, y in generator_images(1, i, ochuman):
            break
        base_dir_custom = "custom_dataset_human_black_background/"
        try:
            os.makedirs(f'{base_dir_custom}')
        except:
            pass
        try:
            os.makedirs(f'{base_dir_custom}features/')
        except:
            pass
        try:
            os.makedirs(f'{base_dir_custom}labels/')
        except:
            pass
        x_name = f"{base_dir_custom}features/{i}_x.jpg"
        y_name = f"{base_dir_custom}labels/{i}_y.jpg"
        cv2.imwrite(x_name, x[0] * 255.)
        cv2.imwrite(y_name, y['seg'][0] * 255.)    

# Data Preparation, espacially masking
# for x, y in generator_images(2, 1):
#     break
# print(x.shape, y['seg'].shape)

# ochuman, image_ids = mask_creation()
# masking_all(ochuman)


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Lambda, GlobalAveragePooling2D, concatenate
from tensorflow.keras.layers import UpSampling2D, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.applications.vgg16 import VGG16
# from keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pickle
import random
from sklearn.model_selection import train_test_split
import datetime
# from google.colab.patches import cv2_imshow

epochs = 5
batch_size = 2
ImgDir = "custom_dataset_human_black_background/"

features = os.listdir(f"{ImgDir}features/")
labels = os.listdir(f"{ImgDir}labels/")
print(len(features), len(labels))

X = features
y = labels

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)

X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.15, random_state=1)

print(len(X_train), len(X_val), len(X_test))

def get_model():
    IMG_HEIGHT = 512
    IMG_WIDTH = 512
    in1 = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3 ))

    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(in1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv4)

    up1 = concatenate([UpSampling2D((2, 2))(conv4), conv3], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up1)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)
    
    up2 = concatenate([UpSampling2D((2, 2))(conv5), conv2], axis=-1)
    conv6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up2)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

    up2 = concatenate([UpSampling2D((2, 2))(conv6), conv1], axis=-1)
    conv7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(up2)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)
    segmentation = Conv2D(3, (1, 1), activation='sigmoid', name='seg')(conv7)

    model = Model(inputs=[in1], outputs=[segmentation])

    losses = {'seg': 'binary_crossentropy'
            }

    metrics = {'seg': ['acc']
                }
    model.compile(optimizer="adam", loss = losses, metrics=metrics)

    return model

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, epoch, logs={}):

        res_dir = "intermediate_results_black_background"

        try:
            os.makedirs(res_dir)
        except:
            print("{res_dir} directory already exist")

        print('Training: epoch {} begins at {}'.format(epoch, datetime.datetime.now().time()))

    def on_epoch_end(self, epoch, logs=None):
        res_dir = "intermediate_results_black_background/"
        print('Training: epoch {} ends at {}'.format(epoch, datetime.datetime.now().time()))
        
        for x_test, y_test in keras_generator_train_val_test(batch_size, choice="test"):
            break
        p = np.reshape(x_test[0], (1, 512, 512, 3))
        prediction = self.model.predict(p)

        x_img = f"{res_dir}{epoch}_X_input.jpg"
        y_img = f"{res_dir}{epoch}_Y_truth.jpg"
        predicted_img = f"{res_dir}{epoch}_Y_predicted.jpg"

        cv2.imwrite(x_img, x_test[0] * 255.)
        cv2.imwrite(y_img, y_test['seg'][0] * 255.)
        cv2.imwrite(predicted_img, prediction[0] * 255.)

def keras_generator_train_val_test(batch_size, choice="train"):
    ImgDir = 'custom_dataset_human_black_background/'
    if choice == "train":
        X = X_train
        y = y_train
    elif choice == "val":
        X = X_val
        y = y_val
    elif choice == "test":
        X = X_test
        y = y_test
    else:
        print("Invalid Option")
        return False
        
    while True:
        x_batch = []
        y_batch = []

        for i in range(batch_size):
            x_rand = random.choice(X)
            y_rand = x_rand[:-5]+"y.jpg"
            
            x_path = f"{ImgDir}features/{x_rand}"
            y_path = f"{ImgDir}labels/{y_rand}"

            x = cv2.imread(x_path)
            y = cv2.imread(y_path)

            x = x / 255.
            y = y / 255.
            
            x_batch.append(x)
            y_batch.append(y)

        
        x_batch = np.array(x_batch)
        # y_batch = np.array(y_batch)

        y_batch = {'seg': np.array(y_batch),
                #    'cls': np.array(classification_list)
                }
        yield x_batch, y_batch


def training_unet():

    # physical_devices = tf.config.list_physical_devices('GPU')
    # for p in physical_devices:
    #     tf.config.experimental.set_memory_growth(p, True)  

    model = get_model()
    model.summary()

    # Training 
    model_name = "models/"+"Unet_black_background.h5"

    modelcheckpoint = ModelCheckpoint(model_name,
                                    monitor='val_loss',
                                    mode='auto',
                                    verbose=1,
                                    save_best_only=True)

    lr_callback = ReduceLROnPlateau(min_lr=0.000001)

    callback_list = [modelcheckpoint, lr_callback, MyCustomCallback()]

    history = model.fit_generator(
    keras_generator_train_val_test(batch_size, choice="train"),
    validation_data = keras_generator_train_val_test(batch_size, choice="val"),
    validation_steps = 1,
    steps_per_epoch=1,
    epochs=epochs,
    verbose=1, 
    shuffle=True,
    callbacks = callback_list,
    )



# Test
# for x, y in keras_generator_train_val_test(2, choice="train"):
#     break
# print(x.shape, y['seg'].shape)
# # cv2_imshow(x[0] * 255.)
# # cv2_imshow(y['seg'][0] * 255.)

# Training 
# training_unet()