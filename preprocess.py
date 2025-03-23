import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import matplotlib

matplotlib.use('Agg')


# ----------------------------- Data Augmentation(Filtration) ----------------------------------
def filteration(image_path):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)])

    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, [224, 224])

    img = img / 255.

    image = tf.expand_dims(img, 0)

    for i in range(9):
        augmented_image = data_augmentation(image)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_image[0])
        plt.axis("off")
        plt.savefig("static/assets/display/augmented.jpg")


# ----------------------------- Resizing (Preprocess) ----------------------------------

def processing(image_path):
    global original_size, resized_shape, res_img, result
    # loading image
    # Getting 3 images_uploaded to work with
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img)
    img = tf.cast(img, tf.float32) / 255.
    original = img
    original_size = img[0].shape
    print('Original size', img[0].shape)

    # setting dim of the resize
    res_img = []
    for i in range(len(img)):
        res = tf.io.read_file(image_path)
        res = tf.io.decode_image(res)
        res = tf.image.resize(res, [224, 224])
        res = tf.cast(res, tf.float32) / 255.
        res_img.append(res)

    # Checcking the size
    resized_shape = res_img[1].shape
    print("RESIZED", res_img[1].shape)

    # Visualizing one of the images_uploaded in the array
    result = res_img[1]
    display_pp(tf.cast(original, tf.float32), tf.cast(result, tf.float32))


# Display one image
def display_one(a, title1="Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.show()


# Display two images_uploaded
def display_pp(a, b):
    title1 = f"Original Size={original_size}"
    title2 = f"Edited Size={resized_shape}"
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(b), plt.title(title2), plt.grid()
    plt.xticks([]), plt.yticks([])
    plt.savefig("static/assets/display/resize.jpg")


# ----------------------------- RGB to HSV (Preprocess) ----------------------------------


# Importing Necessary Libraries
from skimage import data
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt


def hsv(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img)
    img = tf.image.resize(img, [224, 224])
    img = img[:, :, :1]
    image = img / 255.
    plt.subplot(1, 2, 1)
    hsv_image = tf.image.grayscale_to_rgb(img)
    hsv_image = hsv_image / 255.
    display(image, hsv_image)


def display(a, b, title1="Original", title2="Grayscale"):
    plt.subplot(121), plt.imshow(a), plt.title(title1)
    plt.subplot(122), plt.imshow(b), plt.title(title2)
    plt.savefig("static/assets/display/hsv.jpg")


# ----------------------------- Global Color Histogram  ----------------------------------

import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio


def generate_color_histogram(image_path):
    # read original image, in full color
    image = iio.imread(uri=image_path)


    # tuple to select colors of each channel line
    colors = ("red", "green", "blue")

    # create the histogram plot, with three lines, one for
    # each color
    plt.figure()
    plt.xlim([0, 256])
    for channel_id, color in enumerate(colors):
        histogram, bin_edges = np.histogram(
            image[:, :, channel_id], bins=256, range=(0, 256)
        )
        plt.plot(bin_edges[0:-1], histogram, color=color)

    plt.title("Color Histogram")
    plt.xlabel("Color value")
    plt.ylabel("Pixel count")
    plt.savefig("static/assets/display/histogram.jpg")
    plt.close()


# # Example usage
# generate_color_histogram("testing_input/Adenocarcinoma2.png")

# ----------------------------- lbp and clbp  ----------------------------------

import numpy as np
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
import matplotlib.pyplot as plt
import cv2


def plot_lbp_histogram(image_path, save_path, P, R):
    # Load the image from a file
    image = cv2.imread(image_path)

    # Convert the image to grayscale if it is a color image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the LBP or CLBP of the image
    lbp = local_binary_pattern(image, P=P, R=R)

    plt.figure()
    # Plot the LBP or CLBP as a histogram
    plt.hist(lbp.ravel(), bins=256, range=(0, 255), color=np.random.rand(3, ))
    plt.xlabel("Plot")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0, 255, step=32))
    plt.yticks(np.arange(0, 15000, step=2500))
    plt.savefig(save_path)
    plt.close()


# # Plot the LBP histogram for an image
# plot_lbp_histogram("testing_input/img.png", "static/assets/display/lbp.jpg", P=8, R=1)
#
# # Plot the CLBP histogram for the same image
# plot_lbp_histogram("testing_input/img.png", "static/assets/display/clbp.jpg", P=8, R=2)
