### Registration functions for PET reconstruction ###
### Sam Porter 1st version 2023-29-11 ###

import numpy as np
from scipy.ndimage import affine_transform
import sirf.Reg as reg

def affine_transform_2D(theta, tx, ty, sx, sy, image_arr):
    ''' create a random affine transformation for 2D images '''
    # create the transformation matrix
    transformation_matrix = np.array([[sx*np.cos(theta), -sy*np.sin(theta), tx],
                                        [sx*np.sin(theta),  sy*np.cos(theta), ty],
                                        [0, 0, 1]])

    # apply the transformation
    image_arr_transformed = affine_transform(image_arr, transformation_matrix, order=1)
    return image_arr_transformed

def generate_transformed_image(image, attn):
    ''' Generate n_images number of images with affine transformations '''
    # Use some strategy to choose the transformation parameters. Here, using np.random for demonstration
    theta = np.random.uniform(-np.pi/32, np.pi/32)
    tx, ty = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
    sx, sy = np.random.uniform(0.95, 1.05), np.random.uniform(0.95, 1.05)
    
    vol_im = image.as_array()
    vol_transformed = affine_transform_2D(theta, tx, ty, sx, sy, vol_im[0,:])
    vol_transformed = np.expand_dims(vol_transformed, axis=0)
    image_transform = image.clone().fill(vol_transformed)

    vol_attn = attn.as_array()
    vol_attn_transformed = affine_transform_2D(theta, tx, ty, sx, sy, vol_attn[0,:])
    vol_attn_transformed = np.expand_dims(vol_attn_transformed, axis=0)
    attn_transform = attn.clone().fill(vol_attn_transformed)
    return image_transform, attn_transform

def find_transformation_matrix(reference_image, floating_image):
    """ Find the transformation matrices between the reference image and the floating images.
    """
    registration = reg.NiftyAladinSym()
    registration.set_reference_image(reference_image)
    registration.set_floating_image(floating_image)
    registration.set_parameter('SetLevelsToPerform', '4')
    registration.set_parameter('SetMaxIterations', '10')
    registration.process()
    transformation_matrix = registration.get_transformation_matrix_forward()

    return transformation_matrix

def apply_transformation_matrix(transformation_matrix, floating_image, reverse=False):
    """ Apply the transformation matrices to the floating images.
    """
    res = reg.NiftyResampler()
    res.set_floating_image(floating_image)
    res.set_reference_image(floating_image) # this is to ensure the output image has the same size as the input image
    res.set_interpolation_type_to_cubic_spline()
    res.add_transformation(transformation_matrix)
    if reverse:
        registered_image = res.adjoint(floating_image)
    else:
        registered_image = res.direct(floating_image)

    return registered_image