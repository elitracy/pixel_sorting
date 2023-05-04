import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io
import cv2

def pixel_sort(image, mask, sort_property="HUE"):
    hsv = 0

    match sort_property:
        case "HUE":
            hsv = 0
        case "SAT":
            hsv = 1
        case "VAL":
            hsv = 2
        case _:
            hsv = 0

    x = color.rgb2hsv(image/255.)[:,:,hsv] # index either hue saturation or value channel
    mask = mask.astype(bool)

    masked_image = np.ma.masked_where(~mask, x)
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
    mask = cv2.drawContours(mask,contours, -1, (255), thickness=cv2.FILLED) 
    contoured_image = cv2.drawContours(image,contours, -1, (255), thickness=cv2.FILLED)

    mask = mask.astype(bool)

    return mask, contoured_image

def main():
    image = plt.imread("../images/cloud.jpg")
    mask_t = threshhold_mask(np.copy(image), 200)
    mask_c, _ = contour_mask(np.copy(image))

    hue_sorted_image_contour = pixel_sort(image, mask_c, "HUE")
    luminance_sorted_image_contour = pixel_sort(image, mask_c, "VAL")
    saturation_sorted_image_contour = pixel_sort(image, mask_c, "SAT")
    
    hue_sorted_image_threshhold = pixel_sort(image, mask_t, "HUE")
    luminance_sorted_image_threshhold = pixel_sort(image, mask_t, "VAL")
    saturation_sorted_image_threshhold = pixel_sort(image, mask_t, "SAT")
        
    io.imshow_collection([
                            image,
                            mask_c,
                            mask_t,
                                
                            hue_sorted_image_contour,
                            saturation_sorted_image_contour,
                            luminance_sorted_image_contour,

                            hue_sorted_image_threshhold,
                            luminance_sorted_image_threshhold,
                            saturation_sorted_image_threshhold
                        ])
    io.show()

if __name__ == "__main__":
    main()
