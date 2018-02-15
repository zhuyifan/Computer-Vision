# This code is part of:
#
#   CMPSCI 370: Computer Vision, Spring 2018
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji
#
#   Homework 1

import numpy as np

def alignChannels(img, max_shift):
    # raise NotImplementedError("You should implement this.")
    assert (img.shape[2] == 3)
    blue = img[:, :, 0]
    green = img[:, :, 1]
    red = img[:, :, 2]

    shift_i = [0, 0]
    shift_j = [0, 0]

    min_1 = float("inf")
    min_2 = float("inf")

    for i in range (-max_shift[0], max_shift[0]+1):
        for j in range(-max_shift[1], max_shift[1]+1):
            s_1 = np.sum(np.power(np.roll(green, [i, j],axis=[0, 1])-blue, 2))
            s_2 = np.sum(np.power(np.roll(red, [i, j],axis=[0, 1])-blue, 2))

            if s_1 < min_1:
                min_1 = min(min_1, s_1)
                shift_i[0] = i
                shift_j[0] = j
            if s_2 < min_2:
                min_2 = min(min_2, s_2)
                shift_i[1] = i
                shift_j[1] = j

    img[:, :, 1] = np.roll(green, [shift_i[0], shift_j[0]],axis=[0, 1])
    img[:, :, 2] = np.roll(red, [shift_i[1], shift_j[1]],axis=[0, 1])

    pred_shift = np.array([shift_i, shift_j]).T
    return img, pred_shift
