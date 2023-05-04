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

    x = color.rgb2hsv(image)[:,:,hsv] # index either hue saturation or value channel
    mask = mask.astype(bool)

    masked_image = np.ma.masked_where(mask, x)
    pixel_sorted_image = np.copy(image)


    WIDTH, _, CHANNELS = image.shape

    for channel in range(CHANNELS):
        for i in range(WIDTH):
            # find sort masked regions
            sorted_row_mask_indices = np.sort(np.where(masked_image.mask[i]))[0]

            if sorted_row_mask_indices.size > 1:


                # get original masked image pixels
                row = pixel_sorted_image[i,:,channel] 

                sorted_pixels = row[sorted_row_mask_indices]

                # print(sorted_row_mask_indices, sorted_pixels, sorted_pixels.size)
                strip_start = sorted_row_mask_indices[0]
                strip_length = sorted_row_mask_indices.size
                strip_end = strip_start+strip_length


                # sort the masked region of the image by the indices of the sorted mask 
                # sorted_masked_row = np.take_along_axis(row[strip_start:strip_end], sorted_row_mask_indices, axis=0)
                # copy the sorted pixels into the output image using the mask
                pixel_sorted_image[i, strip_start:strip_end, channel] = sorted_pixels


    return pixel_sorted_image

def threshhold_mask(image, threshold):
    threshholded_image = np.zeros_like(image)
    threshholded_image[image > threshold] = True

    return color.rgb2gray(threshholded_image).astype(bool)

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
    image = plt.imread("../images/man.jpg")
    mask_t = threshhold_mask(np.copy(image), 128)
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
