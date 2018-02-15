# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2018
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 1

import numpy as np

def mosaicImage(img):
    ''' Computes the mosaic of an image.

    mosaicImage computes the response of the image under a Bayer filter.

    Args:
        img: NxMx3 numpy array (image).

    Returns:
        NxM image where R, G, B channels are sampled according to RGRG in the
        top left.
    '''

    image_height, image_width, num_channels = img.shape
    assert(num_channels == 3) #Checks if it is a color image

    output = np.zeros((image_height, image_width))

    for i in range(image_height):
        for j in range(image_width):
            if i % 2 == 0 and j % 2 == 0:
                output[i][j] = img[i][j][2]
            elif i % 2 == 0 and j % 2 != 0:
                output[i][j] = img[i][j][1]
            elif i % 2 != 0 and j % 2 == 0:
                output[i][j] = img[i][j][1]
            else:
                output[i][j] = img[i][j][0]

    return output
