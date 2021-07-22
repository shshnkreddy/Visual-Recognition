import cv2
import numpy as np
import imutils
from pywt import dwt2
from PIL import Image, ImageEnhance
from skimage.filters import gabor, gaussian
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scale = 0.5

image = cv2.imread(r'D:/Rishabh_Pant.jpg')
image = cv2.resize(image,(int(scale*image.shape[1]),int(scale*image.shape[0])),interpolation = cv2.INTER_AREA)

def remove_background(image):
    BLUR = 21
    CANNY_THRESH_1 = 10
    CANNY_THRESH_2 = 200
    MASK_DILATE_ITER = 18
    MASK_ERODE_ITER = 20
    MASK_COLOR = (0.0,0.0,1.0) # In BGR format

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    #-- Find contours in edges, sort by area ---------------------------------------------
    contour_info = []
   
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    #-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    #-- Smooth mask, then blur it --------------------------------------------------------
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask]*3)    # Create 3-channel alpha mask

    #-- Blend masked img into MASK_COLOR background --------------------------------------
    mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
    img         = image.astype('float32') / 255.0                 #  for easy blending

    masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR) # Blend
    masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

    #cv2.imshow('img', masked)                                   # Display
    #cv2.waitKey()

    return masked
    

def skin_segmentation(image):
    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 58, 50], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")

    # resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
    image = imutils.resize(image, width = 400)
    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
	# mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(image, image, mask = skinMask)

    # show the skin in the image along with the mask
    #cv2.imshow("images", np.hstack([image, skin]))
    cv2.imshow("image",skin)

def skin_segmentation_YCrCb(image):
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)
    blur = cv2.GaussianBlur(image, (5,5), 0)
    imageYCrCb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)
    skinYCrCb = cv2.bitwise_or(image, image, mask=skinRegionYCrCb)
    cv2.imshow('skinYCrCb', skinYCrCb)

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
    
    #result_im = Image.fromarray(result, mode='L')
    # cv2.imshow("1", result)
    # cv2.imshow("2", result2)
    # cv2.imshow("3",result3)
    rgb = np.dstack((result,result2,result3))
    cv2.imshow("combined",rgb)

image = remove_background(image)    
gabor_filters_segmentation(image)    
    
skin_segmentation_YCrCb(image)
skin_segmentation(image)

cv2.waitKey(0)
cv2.destroyAllWindows()