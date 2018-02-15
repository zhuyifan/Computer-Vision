# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2018
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 1 

import numpy as np

def demosaicImage(image, method):
    ''' Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    '''

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == 'nn':
        return demosaicNN(image.copy()) # Implement this
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    '''Baseline demosaicing.
    
    Replaces missing values with the mean of each color channel.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline 
        algorithm.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[0:image_height:2, 0:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 0] = img[0:image_height:2, 0:image_width:2]

    blue_values = img[1:image_height:2, 1:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 2] = img[1:image_height:2, 1:image_width:2]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img


def demosaicNN(img):
    '''Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of size NxMx3 demosaiced using the nearest neighbor 
        algorithm.
    '''

    height, width = img.shape

    mos_img = np.zeros((height, width, 3))

    for i in range(height):
        for j in range(width):
            if i % 2 == 0 and j % 2 == 0:
                if i == height - 1 and j == width - 1:
                    mos_img[i][j] = [img[i - 1][j - 1], img[i][j - 1], img[i][j]]
                elif i == height - 1:
                    mos_img[i][j] = [img[i - 1][j + 1], img[i][j + 1], img[i][j]]
                elif j == width - 1:
                    mos_img[i][j] = [img[i + 1][j - 1], img[i][j - 1], img[i][j]]
                else:
                    mos_img[i][j] = [img[i + 1][j + 1], img[i][j + 1], img[i][j]]
            elif i % 2 == 0 and j % 2 != 0:
                if i == height - 1:
                    mos_img[i][j] = [img[i - 1][j], img[i][j], img[i][j - 1]]
                else:
                    mos_img[i][j] = [img[i + 1][j], img[i][j], img[i][j - 1]]
            elif i % 2 != 0 and j % 2 == 0:
                if j == width - 1:
                    mos_img[i][j] = [img[i][j - 1], img[i][j], img[i - 1][j]]
                else:
                    mos_img[i][j] = [img[i][j + 1], img[i][j], img[i - 1][j]]
            else:
                mos_img[i][j] = [img[i][j], img[i][j - 1], img[i - 1][j - 1]]

    return mos_img

