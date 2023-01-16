import numpy as np
import operator
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
np.seterr(over='ignore')



datadir="D:\s1\Behloul\TP CS-ALTP\Faces"
path = os.path.join(datadir)
files = os.listdir(path)
X, y = np.reshape(files,(20, 20)), range(20)
#split the data: test = 10% from the dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,  test_size=0.10)


def binaryToDecimal(binary, n):
    decimal = 0
    decimal = decimal + binary * pow(2, n)
    return decimal
# get the center's pixels neighbors
def getNeighborsOfCenter(i, j,width,height,image):

    pixels = []
    x = np.arange(max(i - 1, 0), min(i + 1 + 1, width))
    y = np.arange(max(j - 1, 0), min(j + 1 + 1, height))
    x, y = np.meshgrid(x, y)
    R = np.sqrt((x - i) ** 2 + (y - j) ** 2)  # ecludien
    mask = R != 0
    length = np.size(x[mask])
    for n in range(length):
        l = x[mask][n]
        p = y[mask][n]
        pixels.append(image[l][p])
    return pixels
# get the upper csaltp code
def upper_csaltp (pixels, weberk):
    positivecode = 0
    for n in range(4):
        #weberk = (centre - pixels[n]) / centre
        s = pixels[n] - pixels[n+4] * weberk
        if (pixels[n] - pixels[n +4] > s):
            pixels[n] = 1
            pixels[n] = binaryToDecimal(pixels[n], n)
        else:
            pixels[n] = 0
        positivecode += pixels[n]
    return positivecode

#get the lower csaltp code
def lower_csaltp(pixels, weberk):

    negativecode = 0
    for n in range(4):
        #weberk = (centre - pixels[n]) / centre
        s = pixels[n] - pixels[n+4] * weberk
        if (pixels[n] - pixels[n+4] <= -s):
            pixels[n] = -1
        else:
            pixels[n] = 0
        if (pixels[n] == -1):
            pixels[n] = 1
            pixels[n] = binaryToDecimal(pixels[n], n)
        negativecode += pixels[n]
    return negativecode

#####   """Split a matrix into sub-matrices."""  #######
def split(matrix, nrows, ncols):
    r, h = matrix.shape
    return (matrix.reshape(h//nrows, nrows, -1, ncols)
              .swapaxes(1, 2)
              .reshape(-1, nrows, ncols))

def histogram (cell):
  histo= []
  for i in range ( len(cell)):
    hist, bins = np.histogram(cell[i].ravel(), bins=np.arange(0, 17))
    histo.append(hist)
  return histo

# main csaltp
def cs_Altp(image, weberk):
      l= 0
      width = image.shape[0]
      height = image.shape[1]

      cs_altp_upper = np.zeros((width - 2) * (height - 2))
      cs_altp_lower = np.zeros((width - 2) * (height - 2))
      for i in range(1, image.shape[0]-1 ):
        for j in range(1, image.shape[1]-1 ):

            ### get centre
            centre = image[i][j]
            ### get centre neighbors
            pixels = getNeighborsOfCenter(i,j,width,height, image)
            pi = pixels.copy()
            ### get the cs_altp  positive codes
            cs_altp_upper[l]= upper_csaltp(pixels,weberk)
            #### get the cs_altp negative codes
            cs_altp_lower[l] = lower_csaltp(pi, weberk)
            l = l+ 1
      # create the negative and positive matrix
      upper_matrix = np.reshape(cs_altp_upper, ((width - 2), (height - 2)))
      lower_matrix = np.reshape(cs_altp_lower, ((width - 2), (height - 2)))
      # split the matrix
      a= split(upper_matrix,5,5)
      b= split(lower_matrix,5,5)
      # calculate histograme of both images negative and positive
      histo_pos = np.array(histogram(a))
      histo_neg= np.array(histogram(b))
      # cocatunate the histogrames
      histo_image= np.concatenate((histo_pos,histo_neg) , axis=None).flatten()

      return histo_image

def euclDistance(train_histo,histo_test):
    distances= []
    # instead of writing train_histo[0][x]
    # this way we can get only the histograms on train_histo without the names of train image
    n = 1
    t = [x[n] for x in train_histo]
    for x in range(len(t)):
        dist = np.sqrt(np.sum(np.square(t[x] - histo_test)))
        # distances= the name of the train image + the distance between it and the test image
        distances.append((train_histo[x][0], dist))
    # we sort the distances from the small distance till the big distance
    distances.sort(key=operator.itemgetter(1))
    return distances

# get the nearest neighbors of the test image
def getNN ( distances , k):
    neighbors = []
    rows = 2
    cols = 4
    ax = []
    fig = plt.figure()
    # neighbors = the k nearest images based on their distances
    for x in range(k):
        neighbors.append((distances[x][0]))
    # #we plot the nearest images
    # for a in range(len(neighbors)):
    #     path = "Faces/" + neighbors[a]
    #     img = Image.open(path).convert('L')
    #     ax.append(fig.add_subplot(rows, cols, a + 1))
    #     subplot_title = (neighbors[a] + str(a))
    #     ax[-1].set_title(subplot_title)
    #     plt.imshow(img)
    #     fig.tight_layout()
    # plt.show()
    return neighbors

##### used to calculate the accuracy ######
# def getAccuracy(  distances, testSet , k):
#     # we calculate the accuracy manually, by the show of nearest neighbors for the test image
#     result= []
#     # calculate the accuracy for k = 8 and k= 7 .. k=1
#     while k == 7:
#         getNN(distances, k)
#         correct = int(input("enter number of correct images"))
#         accuracy = (correct / float((len(testSet)) * k)) * 100.0
#         result.append((accuracy, k))
#         k = k - 1
#     return result

def main ():
    train_histo = []
    weberk = float(input("Enter WeberK value : "))
    k = int(input(" enter number of k = "))

    # calculate the histograms of all the train images
    for x in range(len(X_train)):
        for y in range(20):
            path = "Faces/" + X_train[x][y]
            img = Image.open(path).convert('L')
            imgMat = np.array(img)
            # get the histogram of the train image
            histo = cs_Altp(imgMat, weberk)
            # create a matrix of all the histograms (the name of the image + its histogram)
            train_histo.append((X_train[x][y], histo))

    # test images
    for x in range(len(X_test)):
        for y in range(20):
            path = "Faces/" + X_test[x][y]
            img = Image.open(path).convert('L')
            imgMat = np.array(img)
            # calculate the histogram of the test image
            histo_test = cs_Altp(imgMat,weberk)
            # we calculate the distance between the histogram test and all of the histograms train
            distances = euclDistance(train_histo, histo_test)
            # to plot the test image
            # plt.imshow(img)
            # plt.show()
            result = getNN(distances, k)
            print("result= ", result)
            #result = getAccuracy(distances , X_test , k)


main()


