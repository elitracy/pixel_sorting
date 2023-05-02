import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io

def hue_sort(image, mask=None):
    hue = color.rgb2hsv(image/255.)[:,:,0]
    hue_sorted_hsv = hue[np.argsort(hue)]
    hue_sorted_rgb = np.clip(color.hsv2rgb(hue_sorted_hsv)*255, 0, 255)

    return hue_sorted_rgb

def luminance_sort(image, mask):
    luminance = color.rgb2hsv(image/255.)[:,:,2]
    # print(color.rgb2hsv(image).shape)

    sorted_indices = np.argsort(luminance, axis=1)
    masked_luminance = np.where(mask, luminance, np.inf)
    sorted_luminance = np.take_along_axis(masked_luminance, sorted_indices, axis=1)

    return sorted_luminance

def saturation_sort(image, mask=None):
    saturation = color.rgb2hsv(image/255.)[:,:,1]

    saturation_sorted_indices = np.argsort(saturation)
    saturation_sorted_rgb = image[saturation_sorted_indices]

    return saturation_sorted_rgb

def pixel_sort(image, mask, sort_property="HUE", axis=1):

    mask = mask.astype(bool)

    hsv = 0
    if sort_property == "HUE":
        hsv = 0
    elif sort_property == "SAT":
        hsv = 1
    elif sort_property == "VAL":
        hsv = 2

    x = color.rgb2hsv(image/255.)[:,:,hsv]
    # print(color.rgb2hsv(image).shape)

    masked_image = np.ma.masked_where(~mask, x)
    sorted_indices = np.argsort(masked_image, axis)
    # sorted_mask = np.take_along_axis(masked_image, sorted_indices, axis)

    pixel_sorted_image = np.copy(image)
    pixel_sorted_image[masked_image.mask] = 0

    for i in range(masked_image.shape[0]):
        pixel_sorted_image[i, masked_image.mask[i]] = np.take_along_axis(image[i], sorted_indices[i], axis=0)

    return pixel_sorted_image


def threshhold_mask(image):
    mask = image > 128

    threshholded_image = np.zeros_like(image)
    threshholded_image[mask] = 255

    return color.rgb2gray(threshholded_image)

def main():
    image = plt.imread("../images/david.jpg")
    mask = threshhold_mask(image)

    plt.imshow(mask,cmap='gray')
    plt.show()
    WIDTH, HEIGHT, CHANNELS = image.shape

    hue_sorted_image = np.zeros_like(image)
    luminance_sorted_image = np.zeros_like(image)
    saturation_sorted_image = np.zeros_like(image)



    hue_sorted_image = pixel_sort(image, mask, "HUE")
    luminance_sorted_image = pixel_sort(image, mask, "VAL")
    saturation_sorted_image = pixel_sort(image, mask, "SAT")
        
    io.imshow_collection([image, hue_sorted_image, luminance_sorted_image, saturation_sorted_image])
    io.show()

if __name__ == "__main__":
    main()
