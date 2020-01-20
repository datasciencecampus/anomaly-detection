from keras.datasets import mnist, fashion_mnist
from models import load_model
import numpy as np
import os
import argparse
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import tifffile as tiff
import fnmatch,os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split

curdir = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--optimizer', choices=['adam','sgd','adagrad'], default='sgd')
parser.add_argument('--loss', choices=['mean_squared_error', 'binary_crossentropy'], default='mean_squared_error')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--test_samples', type=int, default=50)
parser.add_argument('--model_name', default='convolutional_autoencoder')
parser.add_argument('--result', default=os.path.join(curdir,'loss_vs_samples.png'))

# def imageprepare(argv):
#     """
#     This function returns the pixel values.
#     The imput is a png file location.
#     """
#     print(argv)
#     im = Image.open(argv).convert('L')
#     print('image opened')
#     width = float(im.size[0])
#     height = float(im.size[1])
#     newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

#     if width > height:  # check which dimension is bigger
#         # Width is bigger. Width becomes 20 pixels.
#         nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
#         if (nheight == 0):  # rare case but minimum is 1 pixel
#             nheight = 1
#             # resize and sharpen
#         img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#         wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
#         newImage.paste(img, (4, wtop))  # paste resized image on white canvas
#     else:
#         # Height is bigger. Heigth becomes 20 pixels.
#         nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
#         if (nwidth == 0):  # rare case but minimum is 1 pixel
#             nwidth = 1
#             # resize and sharpen
#         img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#         wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
#         newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

#     #newImage.save(argv)

#     tv = list(newImage.getdata())  # get pixel values

#     # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
#     tva = [(255 - x) * 1.0 / 255.0 for x in tv]
#     #print(tva)
#     return np.array(tva)

def main(args):
    
    def imageprepare(y):
        
        #print('image opened')
        #print(y)
        im = Image.open(y).convert('L')
        
        
        
        width = float(im.size[0])
        height = float(im.size[1])
        newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

        if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
            nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
            if (nheight == 0):  # rare case but minimum is 1 pixel
                nheight = 1
            # resize and sharpen
            img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
            newImage.paste(img, (4, wtop))  # paste resized image on white canvas
        else:
        # Height is bigger. Heigth becomes 20 pixels.
            nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
            if (nwidth == 0):  # rare case but minimum is 1 pixel
                nwidth = 1
            # resize and sharpen
            img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
            wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
            newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
            
        tv = list(newImage.getdata())  # get pixel values

        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
        tva = [(255 - x) * 1.0 / 255.0 for x in tv]
         #print(tva)
        #print('image prepared')
        return np.array(tva)
    
    



    X = []

    for x in train_data:
        # convert color image to 2D array (grayscale) & rescale
        #print(x)
        data=imageprepare(x)#tiff.imread(imageprepare(x))/255.
        #data = tiff.imread(x)/255.0#cv2.imread(x,0) / 255.0
        #img = Image.open(x).convert("L")
        data=data.reshape(28,28)
        #label = 0 # label/class of the image
        X.append(data)

    # loop trough all images ...

    # split for training & testing
    x_train, x_test = train_test_split(X, test_size=0.33)
    
    x_train=np.array(x_train)
    
    x_test=np.array(x_test)
    
    print(np.shape(x_test))

    X = []

    for x in valid_data:
        # convert color image to 2D array (grayscale) & rescale
        #data = tiff.imread(x)/255.0#cv2.imread(x,0) / 255.0
        data=imageprepare(x)#tiff.imread(imageprepare(x))/255.
        data=data.reshape(28,28)
        #label = 0 # label/class of the image
        X.append(data)
        
    print(np.shape(X))

    # loop trough all images ...

    # split for training & testing
    x_abnormal=np.array(X)

    # prepare normal dataset (Mnist)
    #(x_train, _), (x_test, _) = mnist.load_data()
    #x_train = x_train / 255. # normalize into [0,1]
    #x_test = x_test / 255.

    # prapare abnormal dataset (Fashion Mnist)
    #(_, _), (x_abnormal, _) = fashion_mnist.load_data()
    #x_abnormal = x_abnormal / 255.

    # sample args.test_samples images from eaech of x_test and x_abnormal
    perm = np.random.permutation(args.test_samples)
    x_test = np.array(x_test)[perm][:args.test_samples]
    x_abnormal = np.array(x_abnormal)[perm][:args.test_samples]

    # train each model and test their capabilities of anomaly deteciton
   # model_names = ['autoencoder', 'deep_autoencoder', 'convolutional_autoencoder']
    model_names = [args.model_name]
    for model_name in model_names:
        # instantiate model
        model = load_model(model_name)

        # reshape input data according to the model's input tensor
        if model_name == 'convolutional_autoencoder':
            x_train = x_train.reshape(-1,28,28,1)
            x_test = x_test.reshape(-1,28,28,1)
            x_abnormal = x_abnormal.reshape(-1,28,28,1)
        elif model_name == 'autoencoder' or model_name == 'deep_autoencoder':
            x_train = x_train.reshape(-1,28*28)
            x_test = x_test.reshape(-1,28*28)
            x_abnormal = x_abnormal.reshape(-1,28*28)
        else:
            raise ValueError('Unknown model_name %s was given' % model_name)

        # compile model
        model.compile(optimizer=args.optimizer, loss=args.loss)

        # train on only normal training data
        model.fit(
            x=x_train,
            y=x_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # test
        x_concat = np.concatenate([x_test, x_abnormal], axis=0)
        losses = []
        for x in x_concat:
            # compule loss for each test sample
            x = np.expand_dims(x, axis=0)
            loss = model.test_on_batch(x, x)
            losses.append(loss)

        # plot
        #plt.scatter(range(len(losses)), losses, linewidth=1, label=model_name)
        plt.plot(range(len(losses)), losses, '-o', linewidth=1, label=model_name)
        

        # delete model for saving memory
        del model

        # create graph
        plt.legend(loc='best')
        plt.grid()
        #plt.ylim(min(losses),max(losses))
        plt.xlabel('sample index')
        plt.ylabel('loss')
        plt.savefig(args.result)
        plt.show()
        #plt.clf()



'''
    For the given path, get the List of all files in the directory tree 
'''
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles    



if __name__ == '__main__':
    args = parser.parse_args()
    initial_image_dir='./images/docs'
    train_data_dir = initial_image_dir + '/train'
    validation_data_dir = initial_image_dir + '/valid'

    train_size_images = [x for x in fnmatch.filter(os.listdir(train_data_dir),'*.tif')]

    valid_size_images = [x for x in fnmatch.filter(os.listdir(validation_data_dir),'*.tif')]
    
    
    train_data=getListOfFiles('./images/docs/train')#[train_data_dir+'/'+str(x) for x in train_size_images]
    
    train_data=[x for x in train_data if x.endswith('.tif')]
    
    valid_data=getListOfFiles('./images/docs/valid')#[validation_data_dir+'/'+str(x) for x in valid_size_images]
    
    valid_data=[x for x in valid_data if x.endswith('.tif')]
    main(args)



