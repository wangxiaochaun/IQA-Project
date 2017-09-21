import os

import cv2
import numpy as np
from scipy.ndimage.filters import convolve


def split(img, p_height, p_width, p_height_strip=0, p_width_strip=0, dst_path=""):
    '''
    Split the image into patches with the same size.
    :param img: the image to be spitted
    :param p_height: the height of the demanded patch
    :param p_width: the width of the demanded patch
    :param p_height_strip: the vertical pixel shift between two adjacent patches, e.g., strip=-2, where there are
    two-pixel overlapping
    :param p_width_strip: the horizontal pixel shift between two adjacent patches
    :param dst_path: the dir to store the patches
    :return: none
    '''

    height = img.shape[0]
    width = img.shape[1]

    cnt = 1
    n_height = int(height / p_height)
    n_width = int(width / p_width)

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        print("make the dir!")

    for i in range(n_height):
        for j in range(n_width):
            y = p_height * i + p_height_strip
            x = p_width * j + p_width_strip

            patch = img[y:y + p_height, x:x + p_width, :]

            cv2.imwrite(dst_path + '/%d' % cnt + '.jpg', patch)
            cnt += 1


def local_normalization(img):
    '''
    Normalize the intensity image by its local intensity contrast
    :param img: the input intensity image
    :param window_size: the local normalization window
    :return: the normalized intensity image
    '''

    img_size = np.shape(img)
    img_float = img.astype(np.float)

    mean_kernel = np.array([[1. / 9, 1. / 9, 1. / 9],
                            [1. / 9, 1. / 9, 1. / 9],
                            [1. / 9, 1. / 9, 1. / 9]])

    mu = cv2.filter2D(img_float, -1, mean_kernel, borderType=cv2.BORDER_REPLICATE)

    sigma = np.zeros(img_size)
    result = np.zeros(img_size)
    c = 1

    # Processing the body of the image, the result can be directly resized to the raw size, as depicted below
    for i in range(1, img_size[0] - 1):
        for j in range(1, img_size[1] - 1):
            temp = img_float[i - 1: i + 2, j - 1: j + 2]
            sigma[i, j] = np.sqrt(np.sum(np.square(temp - mu[i, j])))
            result[i, j] = (img_float[i, j] - mu[i, j]) / (sigma[i, j] + c)

    # Processing the image boundaries
    for j in range(1, img_size[1] - 1):
        temp = img_float[0: 2, j - 1: j + 2]
        sigma[0, j] = np.sqrt(np.sum(np.square(temp - mu[0, j])))
        result[0, j] = (img_float[0, j] - mu[0, j]) / (sigma[0, j] + c)
        temp = img_float[img_size[0] - 2: img_size[0], j - 1: j + 2]
        sigma[img_size[0] - 1, j] = np.sqrt(np.sum(np.square(temp - mu[img_size[0] - 1, j])))
        result[img_size[0] - 1, j] = (img_float[img_size[0] - 1, j] - mu[img_size[0] - 1, j]) / \
                                     (sigma[img_size[0] - 1, j] + c)

    for i in range(1, img_size[0] - 1):
        temp = img_float[i - 1: i + 2, 0: 2]
        sigma[i, 0] = np.sqrt(np.sum(np.square(temp - mu[i, 0])))
        result[i, 0] = (img_float[i, 0] - mu[i, 0]) / (sigma[i, 0] + c)
        temp = img_float[i - 1: i + 2, img_size[1] - 2: img_size[1]]
        sigma[i, img_size[1] - 1] = np.sqrt(np.sum(np.square(temp - mu[i, img_size[1] - 1])))
        result[i, img_size[1] - 1] = (img_float[i, img_size[1] - 1] - mu[i, img_size[1] - 1]) / \
                                     (sigma[i, img_size[1] - 1] + c)
    # Processing the four vertices
    temp = img_float[0: 2, 0: 2]
    sigma[0, 0] = np.sqrt(np.sum(np.square(temp - mu[0, 0])))
    result[0, 0] = (img_float[0, 0] - mu[0, 0]) / (sigma[0, 0] + c)
    temp = img_float[0: 2, img_size[1] - 2: img_size[1]]
    sigma[0, img_size[1] - 1] = np.sqrt(np.sum(np.square(temp - mu[0, img_size[1] - 1])))
    result[0, img_size[1] - 1] = (img_float[0, img_size[1] - 1] - mu[0, img_size[1] - 1]) / \
                                 (sigma[0, img_size[1] - 1] + c)
    temp = img_float[img_size[0] - 2: img_size[0], 0: 2]
    sigma[img_size[0] - 1, 0] = np.sqrt(np.sum(np.square(temp - mu[img_size[0] - 1, 0])))
    result[img_size[0] - 1, 0] = (img_float[img_size[0] - 1, 0] - mu[img_size[0] - 1, 0]) / \
                                 (sigma[img_size[0] - 1, 0] + c)
    temp = img_float[img_size[0] - 2: img_size[0], img_size[1] - 2: img_size[1]]
    sigma[img_size[0] - 1, img_size[1] - 1] = \
        np.sqrt(np.sum(np.square(temp - mu[img_size[0] - 2: img_size[0], img_size[1] - 2: img_size[1]])))
    result[img_size[0] - 1, img_size[1] - 1] = \
        (img_float[img_size[0] - 1, img_size[1] - 1] - mu[img_size[0] - 1, img_size[1] - 1]) / \
        (sigma[img_size[0] - 1, img_size[1] - 1] + c)

    result_resize = cv2.resize(result, (img_size[1], img_size[0]))
    # The first version of local normalization
    # return result
    # The second version of local normalization
    return result_resize


def local_normalize(img, const=127.0):
    '''
    Lv's method, seems over smoothed
    :param img: input image
    :param const: constant
    :return: the normalized image
    '''
    k = np.float32([1, 4, 6, 4, 1])
    k = np.outer(k, k)
    kern = k / k.sum()
    mu = convolve(img, kern, mode='nearest')
    mu_sq = mu * mu
    im_sq = img * img
    tmp = convolve(im_sq, kern, mode='nearest') - mu_sq
    sigma = np.sqrt(np.abs(tmp))
    structdis = (img - mu) / (sigma + const)

    # Rescale within 0 and 1
    # structdis = (structdis + 3) / 6
    structdis = 2. * structdis / 3.
    return structdis


def img_process(dir_path, series_filename, dst_path=""):
    '''
    Split the images in the dir_path into patches, each image corresponds to a sub_path
    :param dir_path: the path of the source images
    :param series_filename: the filename of the file that stores the image series
    :param dst_path: the root path of the output patches
    :return: None
    '''
    file = open(dir_path + series_filename)
    if not file:
        print("File is not found!")

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        print("make the dir!")
    cnt = 0
    for line in file:
        line = line.strip('\n')
        print(line)
        img = cv2.imread(dir_path + line)
        print("Processing " + line)
        split(img, p_height=32, p_width=32, dst_path=dst_path + "/" + line[:-4])
        cnt += 1
    print("A total of %d" % cnt + " images have been split!")


# Testing codes
input = cv2.imread("data/demo/01_23_02_Book_arrival_A2_8_to_9_60.bmp")

input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

local_normalization(input, [3, 3])

cv2.waitKey(0)
