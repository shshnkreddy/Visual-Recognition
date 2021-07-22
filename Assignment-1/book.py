import cv2   
import numpy as np    
import math
from sklearn.cluster import KMeans 
from scipy.spatial.distance import cdist                                
import matplotlib.pyplot as plt
from pywt import dwt2
from PIL import Image, ImageEnhance
from skimage.filters import gabor, gaussian
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scale = 0.2

image = cv2.imread(r'D:/Bookshelf.jpg')                                                    #Around 25 books present.
image = cv2.resize(image,(int(scale*image.shape[1]),int(scale*image.shape[0])),interpolation = cv2.INTER_AREA)
cv2.imshow('Original', image)

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    Gaussian = cv2.GaussianBlur(image, (7,7), 0)  #7,7
    canny = cv2.Canny(Gaussian, 100, 175)
    return canny


def countours_method(canny, image):
    # construct and apply a closing kernel to 'close' gaps between 'white'
    # pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("Closed", closed)

    # Finding Contours 

    contours, hierarchy = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    total = 0
    for c in contours:
    # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if the approximated contour has four points, then assume that the
        # contour is a book -- a book is a rectangle and thus has four vertices
        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
            total += 1
    
    
    print("Number of Contours found = " + str(total)) 
    cv2.imshow("Output",image)

def dist(l1,l2):
    x1 = l1[0]
    y1 = l1[1]
    x2 = l2[0]
    y2 = l2[1]

    return (x1-x2)**2 + (y1-y2)**2

def Clustering(lst):
    rho = []
    theta = []
    for i in range(len(lst)):
        rho.append(lst[i][0])
        theta.append(lst[i][1])
    
    theta = list(map(lambda x : (x*180/math.pi), theta))

    x = np.array(rho)
    y = np.array(theta)

    X = np.array(list(zip(x, y))).reshape(len(x), 2)
    #Visualizing the data 
    plt.plot() 
    plt.xlim([np.min(x),np.max(x)]) 
    plt.ylim([np.min(y),np.max(y)]) 
    plt.title('Dataset') 
    plt.scatter(x, y) 
    plt.show()

    inertias = []
    distortions = []
    mapping1 = {}
    mapping2 = {}
    K = range(1,len(lst)//2 + 1)

    for k in K: 
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=k).fit(X) 
        kmeanModel.fit(X)  

        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / X.shape[0])   
        inertias.append(kmeanModel.inertia_) 
        
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
                 'euclidean'),axis=1)) / X.shape[0] 
        mapping2[k] = kmeanModel.inertia_ 

    # plt.plot(K, distortions, 'bx-') 
    # plt.xlabel('Values of K') 
    # plt.ylabel('Distortion') 
    # plt.title('The Elbow Method using Distortion') 
    # plt.show() 

    plt.plot(K, inertias, 'bx-') 
    plt.xlabel('Values of K') 
    plt.ylabel('Inertia') 
    plt.title('The Elbow Method using Inertia') 
    plt.show()


def HoughLines(canny, image):
    lines = cv2.HoughLines(canny,1,np.pi/180,130)                                  #hyper parameter
    h_lines = []
    if lines is not None:
        for i in range(0,len(lines)):
            h_lines.append([lines[i][0][0], lines[i][0][1]])
    
    # _min = dist(h_lines[0],h_lines[1])
    # _max = dist(h_lines[0],h_lines[1])
    # for i in range(0,len(h_lines)): 
    #     for j in range(i+1,len(h_lines)):
    #         _min = min(_min,dist(h_lines[i],h_lines[j]))
    #         _max = max(_max, dist(h_lines[i],h_lines[j]))
    # print("min: " + str(_min) + " max: " + str(_max))
    limit = 200               #200

    filtered_lines = []
    img2 = image.copy()
    for i in range(0,len(h_lines)):                                                     #Removes very similar lines
        flag = False
        for j in range(i+1,len(h_lines)):
            if(dist(h_lines[i],h_lines[j]) < limit): flag = True
        if(flag == False): filtered_lines.append(h_lines[i])

    for i in range(0,len(filtered_lines)):
        rho = filtered_lines[i][0]
        theta = filtered_lines[i][1]  
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img2, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    cv2.imshow('img2',img2)   
    print("Number of books found  = " + str(len(filtered_lines)))

    Clustering(filtered_lines)

def get_image_energy(pixels):
    """
    :param pixels: image array
    :return: Energy content of the image
    """
    _, (cH, cV, cD) = dwt2(pixels.T, 'db1')
    energy = (cH ** 2 + cV ** 2 + cD ** 2).sum() / pixels.size
    return energy

def get_energy_density(pixels):
    """
    :param pixels: image array
    :param size: size of the image
    :return: Energy density of the image based on its size
    """
    energy = get_image_energy(pixels)
    energy_density = energy / (pixels.shape[0]*pixels.shape[1])
    return round(energy_density*100,5) # multiplying by 100 because the values are very small

def get_magnitude(response):
    """
    :param response: original gabor response in the form: [real_part, imag_part] 
    :return: the magnitude response for the input gabor response
    """
    magnitude = np.array([np.sqrt(response[0][i][j]**2+response[1][i][j]**2)
                        for i in range(len(response[0])) for j in range(len(response[0][i]))])
    return magnitude

def apply_pca(array):
    """
    :param array: array of shape pXd
    :return: reduced and transformed array of shape dX1
    """
    # apply dimensionality reduction to the input array
    standardized_data = StandardScaler().fit_transform(array)
    pca = PCA(n_components=3)
    pca.fit(standardized_data)
    transformed_data = pca.transform(standardized_data)
    return transformed_data.transpose()

def gabor_filters_segmentation(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = image.shape
    pixels = np.asarray(image, dtype="int32")
    energy_density = get_energy_density(pixels)
    # get fixed bandwidth using energy density
    bandwidth = abs(0.4*energy_density - 0.5)
    
    magnitude_dict = {}
    for theta in np.arange(0, np.pi, np.pi / 6):
        for freq in np.array([1.4142135623730951, 2.414213562373095, 2.8284271247461903, 3.414213562373095]): 
            filt_real, filt_imag = gabor(image, frequency=freq, bandwidth=bandwidth, theta=theta)
            # get magnitude response
            magnitude = get_magnitude([filt_real, filt_imag])
            magnitude_dict[(theta, freq)] = magnitude.reshape(image.size)
    
    # apply gaussian smoothing
    gabor_mag = []
    for key, values in magnitude_dict.items():
        # the value of sigma is chosen to be half of the applied frequency
        sigma = 0.5*key[1]
        smoothed = gaussian(values, sigma = sigma)
        gabor_mag.append(smoothed)
    gabor_mag = np.array(gabor_mag)

    # reshape so that we can apply PCA
    value = gabor_mag.reshape((-1, image_size[0]*image_size[1]))

    # get dimensionally reduced image
    pcaed = apply_pca(value.T).astype(np.uint8)
    #print(np.shape(pcaed))
    result = pcaed[0].reshape((image_size[0], image_size[1]))
    result2 = pcaed[1].reshape((image_size[0], image_size[1]))
    result3 = pcaed[2].reshape((image_size[0], image_size[1]))
    
    rgb = np.dstack((result,result2,result3))
    cv2.imshow("combined",rgb)


countours_method(preprocess(image), image)
HoughLines(preprocess(image), image)
gabor_filters_segmentation(image)

cv2.waitKey(0)
cv2.destroyAllWindows()