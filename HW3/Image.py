import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def combine_for_display(ims):
    # todo combine the 5 x 4 images
    # for i in ims.shape[0]:
    #     for j in ims.shape[1]:
    # #         add slice
    # print_image(np.stack(ims, axis=1))
    return 1


def arr_to_im(arr, cols):
    # // is floor division
    return arr.reshape(arr.size // cols, cols)


def print_list(ims):
    im = combine_for_display(ims)
    i = Image.fromarray((im * 255).astype('uint8'), mode='L')
    i.show()


def print_image(im):
    im = arr_to_im(im, 28)
    i = Image.fromarray((im * 255).astype('uint8'), mode='L')
    i.show()


def plot_images(im_list, rows, columns):
    if rows * columns != len(im_list):
        print('Image List size did not match rows x columns size.')
    for i in range(20):  # rows x col
        plt.subplot(rows, columns, i + 1)
        if im_list[i][2] == -1:
            plt.title('4')
        else:
            plt.title('9')
        plt.imshow(im_list[i][1].reshape(28,28), interpolation='nearest')
    plt.show()
