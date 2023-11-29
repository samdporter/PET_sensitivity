import sirf.STIR as stir
import numpy as np

def get_acquisition_model(image, template, attn_image, num_subsets=1):
    acq_model_matrix  = stir.RayTracingMatrix() # create a ray tracing matrix (i.e a system matrix)
    acq_model_matrix.set_num_tangential_LORs(5) # set number of LORs per sinogram bin
    acq_model = stir.AcquisitionModelUsingMatrix(acq_model_matrix) # create acquisition model based on the system matrix
    acq_model_for_attn = stir.AcquisitionModelUsingRayTracingMatrix() # create another acquisition model for use with the acquisition sensitivity model - this is the first three lines of the previous block, but with one line removed
    asm_attn = stir.AcquisitionSensitivityModel(attn_image, acq_model_for_attn) # create acquisition sensitivity model
    acq_model.set_acquisition_sensitivity(asm_attn) # add this acquisition sensitivity model to the acquisition model
    acq_model.num_subsets = num_subsets # set number of subsets
    acq_model.set_up(template,image) # set up the acquisition model
    return acq_model

def add_noise(proj_data, noise_factor, seed):
    """Add Poission noise to acquisition data."""
    proj_data_arr = proj_data.as_array() / noise_factor # Divide by noise factor to be able to scale the amount of noise added
    np.random.seed(seed)
    noisy_proj_data_arr = np.random.poisson(proj_data_arr).astype('float32') # Add Poisson noise
    noisy_proj_data = proj_data.clone() # Create a copy of the acquisition data
    noisy_proj_data.fill(noisy_proj_data_arr*noise_factor)  # Fill the copy with the noisy data
    return noisy_proj_data

def get_acquired_data(image, acq_model, noise_factor=0.1, seed=50):
    """Get acquisition data from image, add noise and return."""
    acquired_data=acq_model.forward(image) # create acquisition data by forward projection of the image
    noisy_data = add_noise(acquired_data, noise_factor, seed) # add noise to the acquisition data
    return noisy_data