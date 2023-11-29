import numpy as np

def crop_image_to_circle(image, radius):
    """ Crop the image to a circle of a given radius.
    """
    image_arr = image.as_array()[0,:] # get the image array
    # create circular mask
    mask = np.zeros_like(image_arr)
    x, y = np.ogrid[:image_arr.shape[0], :image_arr.shape[1]]
    cx, cy = image_arr.shape[0]//2, image_arr.shape[1]//2
    mask_area = (x - cx)**2 + (y - cy)**2 <= radius**2
    mask[mask_area] = 1
    # apply mask
    image_arr[mask==0] = 0
    image.fill(np.expand_dims(image_arr, axis=0))


def division(a, b, eps=1e-10):
    return (a+eps) / (b + eps)