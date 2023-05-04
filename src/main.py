import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import cv2
import random
import os

def random_pixel_sort(image, mask, min_strip_length: int, max_strip_length: int, sort_property="HUE"):
    _, HEIGHT, _ = image.shape

    assert(max_strip_length < HEIGHT)
    assert(min_strip_length >= 0)

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

    mask = mask.astype(bool)
    rows, _ = np.where(mask)
    mask_top, mask_bottom = rows.min(), rows.max()

    pixel_sorted_image = np.copy(image)
    hsv_image = color.rgb2hsv(pixel_sorted_image)[:,:,hsv]

    for i in range(mask_top, mask_bottom+1):
        strip_length =  random.randint(min_strip_length, max_strip_length)

        mask_indices = np.where(mask[i])[0]

        if mask_indices.size > 0:
            strip_start = mask_indices[0]
            strip_end = strip_start + strip_length
            if strip_end > HEIGHT:
                strip_end = HEIGHT-1
                strip_start = strip_end - strip_length

            sorted_indices = np.argsort(hsv_image[i,strip_start:strip_end])

            # copy the sorted pixels into the output image using the mask
            pixel_sorted_image[i, strip_start:strip_end] = pixel_sorted_image[i, sorted_indices] 

    return pixel_sorted_image

def smart_pixel_sort(image, mask, sort_property="HUE"):
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

    image_name = 'space'
    file_path = '../images/' + image_name + '.jpg'
    image = plt.imread(file_path)

    output_path = '../results/' + image_name + "/"
    if not os.path.exists(output_path):
        os.makedirs('../results/' + image_name + "/")


    _, HEIGHT, _ = image.shape

    MIN_SORT_LENGTH = int(HEIGHT * 1/8)
    MAX_SORT_LENGTH = int(HEIGHT * 1/4)
    THRESHOLD = 128
    

    t_mask  = threshhold_mask(np.copy(image), THRESHOLD)
    c_mask, c_mask_layer = contour_mask(np.copy(image))

    plt.imsave(output_path+"threshold_mask_"+image_name+".jpg", t_mask, cmap='gray')
    plt.imsave(output_path+"contour_mask_"+image_name+".jpg", c_mask_layer)


    # =================== RANDOM SORT | CONTOUR ===================
    hue_random_sorted_image_contour = random_pixel_sort(image, c_mask, MIN_SORT_LENGTH, MAX_SORT_LENGTH, "HUE")
    value_random_sorted_image_contour = random_pixel_sort(image, c_mask, MIN_SORT_LENGTH, MAX_SORT_LENGTH, "VAL")
    saturation_random_sorted_image_contour = random_pixel_sort(image, c_mask, MIN_SORT_LENGTH, MAX_SORT_LENGTH , "SAT")

    plt.imsave(output_path+"hue_random_contour_"+image_name+".jpg", hue_random_sorted_image_contour)
    plt.imsave(output_path+"value_random_contour_"+image_name+".jpg",value_random_sorted_image_contour)
    plt.imsave(output_path+"satuation_random_contour_"+image_name+".jpg",saturation_random_sorted_image_contour)
     
    # =================== RANDOM SORT | THRESHOLD ===================
    hue_random_sorted_image_threshhold = random_pixel_sort(image, t_mask, MIN_SORT_LENGTH, MAX_SORT_LENGTH, "HUE")
    value_random_sorted_image_threshold = random_pixel_sort(image, t_mask, MIN_SORT_LENGTH, MAX_SORT_LENGTH, "VAL")
    saturation_random_sorted_image_threshold = random_pixel_sort(image, t_mask, MIN_SORT_LENGTH, MAX_SORT_LENGTH , "SAT")

    plt.imsave(output_path+"hue_random_threshold_"+image_name+".jpg",hue_random_sorted_image_threshhold)
    plt.imsave(output_path+"value_random_threshold_"+image_name+".jpg",value_random_sorted_image_threshold)
    plt.imsave(output_path+"saturation_random_threshold_"+image_name+".jpg",saturation_random_sorted_image_threshold)

    # =================== SMART SORT | CONTOUR ===================
    hue_smart_sorted_image_contour = smart_pixel_sort(image, c_mask, "HUE")
    value_smart_sorted_image_contour = smart_pixel_sort(image, c_mask, "VAL")
    saturation_smart_sorted_image_contour = smart_pixel_sort(image, c_mask, "SAT")

    plt.imsave(output_path+"hue_smart_contour_"+image_name+".jpg",hue_smart_sorted_image_contour)
    plt.imsave(output_path+"value_smart_contour_"+image_name+".jpg",value_smart_sorted_image_contour)
    plt.imsave(output_path+"saturation_smart_contour_"+image_name+".jpg",saturation_smart_sorted_image_contour)
     
    # =================== SMART SORT | THRESHOLD ===================
    hue_smart_sorted_image_threshhold = smart_pixel_sort(image, t_mask, "HUE")
    value_smart_sorted_image_threshold = smart_pixel_sort(image, t_mask, "VAL")
    saturation_smart_sorted_image_threshold = smart_pixel_sort(image, t_mask, "SAT")

    plt.imsave(output_path+"hue_smart_threshold_"+image_name+".jpg",hue_smart_sorted_image_threshhold)
    plt.imsave(output_path+"value_smart_threshold_"+image_name+".jpg",value_smart_sorted_image_threshold)
    plt.imsave(output_path+"saturation_smart_threshold_"+image_name+".jpg",saturation_smart_sorted_image_threshold)

if __name__ == "__main__":
    main()
