import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io
import cv2

def pixel_sort(image, mask, sort_property="HUE"):

    hsv = 0
    if sort_property == "HUE":
        hsv = 0
    elif sort_property == "SAT":
        hsv = 1
    elif sort_property == "VAL":
        hsv = 2

    x = color.rgb2hsv(image/255.)[:,:,hsv]

    masked_image = np.ma.masked_where(~mask.astype(bool), x)
    sorted_indices = np.argsort(masked_image, axis=1)

    pixel_sorted_image = np.copy(image)
    pixel_sorted_image[masked_image.mask] = 0

    WIDTH, _, CHANNELS = image.shape

    for channel in range(CHANNELS):
        for i in range(WIDTH):
            pixel_sorted_image[i,:,channel] = np.take_along_axis(image[i,:,channel], sorted_indices[i], axis=0)

    return pixel_sorted_image

def threshhold_mask(image, threshhold):
    mask = threshhold > 128

    threshholded_image = np.zeros_like(image)
    threshholded_image[mask] = 255

    return color.rgb2gray(threshholded_image)

def contour_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 128, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros_like(gray)
    mask = cv2.drawContours(mask,contours, -1, (255), 10) 
    contoured_image = cv2.drawContours(image,contours, -1, (255), 10)

    return mask, contoured_image

def main():
    image = plt.imread("../images/david.jpg")
    mask_t = threshhold_mask(np.copy(image), 200)
    mask_c, contours = contour_mask(np.copy(image))

    hue_sorted_image_contour = pixel_sort(image, mask_c, "HUE")
    luminance_sorted_image_contour = pixel_sort(image, mask_c, "VAL")
    saturation_sorted_image_contour = pixel_sort(image, mask_c, "SAT")
    
    hue_sorted_image_threshhold = pixel_sort(image, mask_t, "HUE")
    luminance_sorted_image_threshhold = pixel_sort(image, mask_t, "VAL")
    saturation_sorted_image_threshhold = pixel_sort(image, mask_t, "SAT")
        
    io.imshow_collection([contours,
                          mask_t,
                          np.clip(np.bitwise_and(hue_sorted_image_contour,image),0,255),
                          np.clip(np.bitwise_and(luminance_sorted_image_contour,image),0,255),
                          np.clip(np.bitwise_and(saturation_sorted_image_contour,image),0,255),


                          np.clip(np.bitwise_and(hue_sorted_image_threshhold,image),0,255),
                          np.clip(np.bitwise_and(luminance_sorted_image_threshhold,image),0,255),
                          np.clip(np.bitwise_and(saturation_sorted_image_threshhold,image),0,255)
                         ])
    io.show()

if __name__ == "__main__":
    main()
