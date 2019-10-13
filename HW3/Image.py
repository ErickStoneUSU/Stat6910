import numpy as np
from PIL import Image


def combine_for_display(ims):
    # todo combine the 5 x 4 images
    # for i in ims.shape[0]:
    #     for j in ims.shape[1]:
    # #         add slice
    # print_image(np.stack(ims, axis=1))
    return 1


def arr_to_im(arr, cols):
    # // is floor division
    return arr.reshape(len(arr)//cols, cols)


def print_list(ims):
    im = combine_for_display(ims)
    i = Image.fromarray((im * 255).astype('uint8'), mode='L')
    i.show()


def print_image(im):
    im = arr_to_im(im, 28)
    i = Image.fromarray((im * 255).astype('uint8'), mode='L')
    i.show()
