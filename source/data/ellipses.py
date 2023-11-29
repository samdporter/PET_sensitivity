import numpy as np
import torch
from sirf.STIR import AcquisitionSensitivityModel, AcquisitionModelUsingRayTracingMatrix
from misc import random_phantom, affine_transform_2D_image

MAX_PHANTOM_INTENSITY = 0.096 * 2

def generate_random_transform_values() -> tuple:
    """
    Generates random values for affine transformation.
    """
    theta = np.random.uniform(-np.pi/16, np.pi/16)
    tx, ty = np.random.uniform(-1, 1), np.random.uniform(-1, 5)
    sx, sy = np.random.uniform(0.95, 1.05), np.random.uniform(0.95, 1.05)
    return theta, tx, ty, sx, sy

def make_max_n(image: np.ndarray, n: float) -> np.ndarray:
    """
    Scales the image so that its maximum is n.
    """
    image *= n / image.max()
    return image

class EllipsesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for simulated ellipses.

    Parameters:
        radon_transform: SIRF acquisition model used as the forward operator.
        attenuation_image_template: SIRF image data for the template.
        sinogram_template: Template for sinogram data.
        attenuation: Boolean flag for attenuation.
        no_att_sens: Boolean flag for sensitivity without attenuation.
        num_samples: Number of samples in the dataset.
        mode: Dataset mode (train, validation, test).
        seed: Random seed for phantom generation.
    """

    def __init__(self, attenuation_image_template, sinogram_template, num_samples,
                 generate_non_attenuated_sensitivity=False):
                 
        self.num_samples = num_samples

        self.radon_transform = AcquisitionModelUsingRayTracingMatrix()
        self.radon_transform.set_up(sinogram_template, attenuation_image_template)

        self.tmp_acq_model = AcquisitionModelUsingRayTracingMatrix()

        self.attenuation_image_template = attenuation_image_template.clone()

        self.template = sinogram_template
        self.one_sino = sinogram_template.get_uniform_copy(1)

        self.generate_non_attenuated_sensitivity = generate_non_attenuated_sensitivity

    def _get_sensitivity_image(self, ct_image, attenuation=True):
        """
        Generates the sensitivity image.
        """        
        if attenuation:
            asm_attn = AcquisitionSensitivityModel(ct_image, self.radon_transform)
            asm_attn.set_up(self.template)
            self.tmp_acq_model.set_acquisition_sensitivity(asm_attn)
        else:
            asm_attn = AcquisitionSensitivityModel(ct_image.get_uniform_copy(0), self.radon_transform)
            asm_attn.set_up(self.template)
            self.tmp_acq_model.set_acquisition_sensitivity(asm_attn)
            
        self.tmp_acq_model.set_up(self.template, ct_image)
        
        return self.tmp_acq_model.backward(self.one_sino)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Generates one sample of data.
        """
        random_phantom_array = make_max_n(random_phantom(self.attenuation_image_template.shape, 20), MAX_PHANTOM_INTENSITY)
        ct_image = self.attenuation_image_template.clone().fill(random_phantom_array)
        sens_image = self._get_sensitivity_image(ct_image)
        
        theta, tx, ty, sx, sy = generate_random_transform_values()
        ct_image_transform = affine_transform_2D_image(theta, tx, ty, sx, sy, ct_image)
        ct_image_transform.move_to_scanner_centre(self.template)
        sens_image_transform = self._get_sensitivity_image(ct_image_transform)

        if self.generate_non_attenuated_sensitivity:
            sens_image_no_att = self._get_sensitivity_image(ct_image, attenuation=False)
            return np.array([sens_image.as_array().squeeze(), sens_image_no_att.as_array().squeeze(), ct_image_transform.as_array().squeeze()]), sens_image_transform.as_array().squeeze()

        return np.array([sens_image.as_array().squeeze(), ct_image_transform.as_array().squeeze()]), sens_image_transform.as_array().squeeze()
