### OSEM Reconstruction ###
### Includes functions for performing OSEM reconstruction with or without resampling ###
### Sam Porter 1st version 2023-29-11 ###

import sirf.Reg as reg
import sirf.STIR as stir
from misc import crop_image_to_circle, division


def osem_step(input_image, radon_transform, acquired_data, sensitivity_image, transform_matrix=None):
    """
    Perform one OSEM step with or without resampling.

    Args:
        input_image (ImageData): The current estimate of the reconstructed image.
        radon_transform (AcquisitionModel): The Radon transform (forward projection) model.
        acquired_data (AcquisitionData): The acquired projection data.
        sensitivity_image (ImageData): The sensitivity image.
        transform_matrix (TransformationMatrix, optional): Transformation matrix for resampling.

    Returns:
        ImageData: The updated image after one OSEM step.
    """

    # Determine if resampling is needed
    resample_required = transform_matrix is not None

    if resample_required:
        # Initialize resampler
        resample = reg.NiftyResampler()
        resample.add_transformation(transform_matrix)
        resample.set_interpolation_type_to_cubic_spline()
        resample.set_reference_image(input_image)
        resample.set_floating_image(input_image)

        # Apply resampling
        resampled_image = resample.direct(input_image)  # R (u)
        resampled_image.maximum(0)
        processed_image = radon_transform.forward(resampled_image)  # P (R u)
    else:
        processed_image = radon_transform.forward(input_image)  # P (u)

    ratio = acquired_data.clone().fill(division(acquired_data, processed_image))  # f/(P R u) or f/(P u)

    back_projected_ratio = radon_transform.backward(ratio)
    if resample_required:
        back_projected_ratio = resample.adjoint(back_projected_ratio)

    update = division(back_projected_ratio, resample.adjoint(sensitivity_image) if resample_required else sensitivity_image)

    return input_image * update


def osem(input_image, radon_transform, acquired_data, num_subsets, epochs, sensitivity_images, transform_matrices=None, resample=True):
    """
    Perform OSEM reconstruction.

    Args:
        input_image (ImageData): Initial estimate of the reconstructed image.
        radon_transform (AcquisitionModel): The Radon transform (forward projection) model.
        acquired_data (list): List of acquired projection data for each subset.
        num_subsets (int): Number of subsets for OSEM.
        epochs (int): Number of iterations over all subsets.
        sensitivity_images (list): List of sensitivity images for each subset.
        transform_matrices (list, optional): List of transformation matrices for resampling.
        resample (bool): Flag to indicate if resampling is to be used.

    Returns:
        ImageData: The reconstructed image after OSEM iterations.
    """

    if resample and not transform_matrices:
        raise ValueError('Transform matrices must be provided if resampling is True.')

    # Clone the input image to avoid modifying the original
    output_image = input_image.clone()
    crop_image_to_circle(output_image, 60)

    for epoch in range(epochs):
        for subset_num in range(num_subsets):
            radon_transform.subset_num = subset_num

            # Determine if resampling is needed for the current subset
            transform_matrix = transform_matrices[subset_num] if resample else None

            # Perform the OSEM step (with or without resampling)
            output_image = osem_step(output_image, radon_transform, acquired_data[subset_num],
                                     sensitivity_images[subset_num], transform_matrix)

            # Crop the image to a circle
            crop_image_to_circle(output_image, 60)

    return output_image

