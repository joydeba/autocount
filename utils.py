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
    transform = transforms.Compose([transforms.Resize([int(320), int(480)]),
                                transforms.ToTensor()])
    # load the training and test datasets
    # train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    train_data = datasets.ImageFolder( root="dataset", transform=transform)
    # test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    test_data = datasets.ImageFolder( root="dataset", transform=transform)
    # Create training and test dataloaders
    num_workers = 0
    # how many samples per batch to load
    
    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)     
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
    n_epochs = 50



    for epoch in range(1, n_epochs+1):
        # Monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            
            ## add random noise to the input images
            # noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            noisy_imgs = images.clone()

            for idx, image in enumerate(noisy_imgs):
                # image[image <= 0.39] = 0
                image = segmentation_with_masking(255*image.permute(1, 2, 0).numpy())  
                noisy_imgs[idx] = image
              
            

            # Clip the images to be between 0 and 1
            noisy_imgs = np.clip(noisy_imgs, 0., 1.)
                    
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
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    # add noise to the test images
    # noisy_imgs = images + noise_factor * torch.randn(*images.shape)
    # noisy_imgs = images
    noisy_imgs = images.clone()
    # for idx, image in enumerate(noisy_imgs):
    #     image[image <= 0.39] = 0
    #     noisy_imgs[idx] = image  

    noisy_imgs = np.clip(noisy_imgs, 0., 1.)
    # get sample outputs
    output = model(noisy_imgs)
    # prep images for display
    noisy_imgs = noisy_imgs.numpy()
    # output is resized into a batch of iages
    output = output.view(batch_size, 3, 320, 480)
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
    # Testing
    testing_model(model, test_loader)


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
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)

    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    # mask = np.zeros((256,256), np.uint8)
    mask = np.zeros(edges.shape[:2], dtype=np.uint8)
    masked = cv2.drawContours(mask, [cnt],-1, 255, -1)
    dst = cv2.bitwise_and(img, img, mask=mask)
    segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return segmented        

# data_grouping_COCO()    
# data_grouping_GWHD()
# data_grouping_BSR()
# show_image_withAnnotation()
denoising_autoencoder()
# segmentation_with_masking_all()
