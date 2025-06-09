#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script generates 3D brain phantom images based on the BrainWeb dataset,
including simulated PET emission (FDG) and attenuation (mu-map) data.
It can downsample the final images by a specified zoom factor.

The script uses the SIRF library for image manipulation and processing.

Example usage:
python your_script_name.py --subject 0 --outres MR --zoom 0.5 --output_emission fdg_zoomed.hv --output_mu mu_zoomed.hv --save_plot plot.png --save_labels brainweb_labels --seed 1337
"""

import argparse
import os
import numpy as np
from tqdm import tqdm
import brainweb

# SIRF imports
import sirf.STIR as pet
import sirf.Reg as reg

def save_nii(im, fname):
    """Save as nii."""
    reg.ImageData(im).write(fname)

def setup_brainweb_labels():
    """
    Fix for wrong label values in the brainweb library.
    See: https://github.com/casperdcl/brainweb/issues/18
    """
    brainweb.Act.marrow = 177
    brainweb.Act.dura = 161
    brainweb.Act.aroundFat = 145


def get_brainweb_labels_as_pet(outres='brainweb', subject=0):
    """
    Downloads the BrainWeb anatomical model and converts it to a SIRF PET ImageData object.
    
    Args:
        outres (str): A string specifying the output voxel size ('mMR', 'MR', or 'brainweb').

    Returns:
        pet.ImageData: The 3D label image as a SIRF object.
    """
    fname, url = sorted(brainweb.utils.LINKS.items())[subject]
    brainweb.get_file(fname, url, ".")
    data = brainweb.load_file(fname)

    res = getattr(brainweb.Res, outres)
    # Pad to a more standard PET size
    new_shape = (data.shape[0], 512, 512)
    padLR, padR = divmod((np.array(new_shape) - data.shape), 2)
    data = np.pad(data, [(p, p + r) for (p, r) in zip(padLR.astype(int), padR.astype(int))], mode="constant")

    return get_as_pet_im(data, res)


def get_as_pet_im(arr, res):
    """
    Converts a NumPy array into a SIRF PET ImageData object.

    Args:
        arr (np.ndarray): The input image data.
        res (tuple): The voxel sizes (z, y, x).

    Returns:
        pet.ImageData: The resulting SIRF image.
    """
    im = pet.ImageData()
    im.initialise(arr.shape, tuple(res))
    im.fill(arr)
    return im


def weighted_add(out, values, weights):
    """
    Performs a weighted addition: out = out + sum(weights[i] * values[i]).

    Args:
        out (pet.ImageData): The image to which the result is added.
        values (list of pet.ImageData): The list of images to add.
        weights (list of float): The list of weights.
    """
    for (w, v) in zip(weights, values):
        out += w * v


def brainweb_labels_to_4d(brainweb_labels_3d, labels, output_prefix=None):
    """
    Takes a 3D image with BrainWeb labels and splits it into a list of 3D masks, one for each label.

    Args:
        brainweb_labels_3d (pet.ImageData): The 3D image with integer labels.
        labels (list of str): List of label names to extract.
        output_prefix (str): Optional prefix to save each label mask to a file.

    Returns:
        list of pet.ImageData: A list of 3D binary mask images.
    """
    all_masks = []
    brainweb_labels_array = brainweb_labels_3d.as_array()
    
    for label in tqdm(labels, desc="Creating label masks"):
        filename = output_prefix + label + ".nii"
        
        if output_prefix and os.path.isfile(filename):
            print(f"Reading {filename}")
            mask = pet.ImageData(filename)
        else:
            value = getattr(brainweb.Act, label)
            mask = brainweb_labels_3d.allocate()
            mask.fill(brainweb_labels_array == value)
            if output_prefix:
                save_nii(mask, filename)

        all_masks.append(mask)

    return all_masks


def get_image_from_labels(all_label_images, label_names, activity_class):
    """
    Generates a quantitative image (e.g., FDG or mu-map) from label masks and activity values.

    Args:
        all_label_images (list of pet.ImageData): List of label mask images.
        label_names (list of str): The names of the labels corresponding to the masks.
        activity_class: The BrainWeb class (e.g., brainweb.FDG, brainweb.Mu) containing activity values.

    Returns:
        pet.ImageData: The final quantitative image.
    """
    if activity_class == brainweb.Mu:
        # Special handling for mu-map values
        bone_value = getattr(activity_class, "bone")
        tissue_value = getattr(activity_class, "tissue")
        all_values = [0, tissue_value, tissue_value, tissue_value, tissue_value, tissue_value, tissue_value,
                      bone_value, tissue_value, tissue_value, bone_value, bone_value]
    else:
        all_values = [getattr(activity_class, l) for l in label_names]

    if len(all_label_images) != len(all_values):
        raise ValueError(f"Number of label images ({len(all_label_images)}) does not match number of values ({len(all_values)}).")

    print(f"Creating image with values for {activity_class.__name__}: {all_values}")
    
    # Initialize output image
    out = all_label_images[0].clone() * all_values[0]
    
    # Add the remaining weighted label images
    weighted_add(out, all_label_images[1:], all_values[1:])
    return out


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Generate BrainWeb phantoms with vessel inserts and zoom.")
    parser.add_argument("--subject", type=int, default=0,
                        help="BrainWeb subject number (0-11).")
    parser.add_argument("--outres", type=str, default="brainweb", choices=['brainweb', 'MR', 'mMR'],
                        help="Voxel size resolution to use for the initial phantom.")
    parser.add_argument("--zoom", type=float, default=0.5,
                        help="Isotropic zoom factor to downsample the final images.")
    parser.add_argument("--output_emission", type=str, default="fdg_zoomed.hv",
                        help="Filename for the output zoomed emission (FDG) image.")
    parser.add_argument("--output_mu", type=str, default="mu_zoomed.hv",
                        help="Filename for the output zoomed mu-map image.")
    parser.add_argument("--save_plot", type=str, default=None,
                        help="If specified, saves a PNG plot of the central slices to this filename.")
    parser.add_argument("--save_labels", type=str, default=None,
                        help="If specified, saves the BrainWeb labels to this filename.")
    parser.add_argument("--seed", type=int, default=1337,)
    
    args = parser.parse_args()

    # --- 1. Initial Setup ---
    setup_brainweb_labels()
    brainweb.seed(args.seed)
    brainweb_label_prefix = "" if args.save_labels is None else args.save_labels + "_"

    # --- 3. Load Base BrainWeb Phantom ---
    print(f"Getting BrainWeb labels with '{args.outres}' resolution...")
    bw_labels_img = get_brainweb_labels_as_pet(args.outres, args.subject)

    # --- 5. Generate Label Masks and Combine with Vessels ---
    # For Emission (FDG)
    fdg_labels = brainweb.FDG.attrs
    all_label_images = brainweb_labels_to_4d(bw_labels_img, fdg_labels, output_prefix=brainweb_label_prefix)

    # For Attenuation (Mu)
    mu_labels = brainweb.Mu.all_labels
    mu_all_label_images = brainweb_labels_to_4d(bw_labels_img, mu_labels, output_prefix=brainweb_label_prefix)

    # --- 6. Create Quantitative Images ---
    # Create emission image (FDG)
    emission_img = get_image_from_labels(all_label_images, fdg_labels, brainweb.FDG)
    
    # Create attenuation image (mu-map)
    mu_img = get_image_from_labels(mu_all_label_images, mu_labels, brainweb.Mu)

    # --- 7. Zoom Images ---
    zoom_factor = args.zoom
    zooms = (zoom_factor, zoom_factor, zoom_factor)
    
    # Calculate new size after zooming
    original_shape = np.array(emission_img.shape)
    new_size = tuple(np.int32(original_shape * zoom_factor))

    print(f"Zooming images by a factor of {zoom_factor} from {original_shape} to {new_size}...")
    
    emission_zoomed = emission_img.zoom_image(
        zooms=zooms, 
        size=new_size, 
        scaling='preserve_values'
    )
    mu_zoomed = mu_img.zoom_image(
        zooms=zooms, 
        size=new_size,
        scaling='preserve_values'
    )

    # --- 8. Save Outputs ---
    emission_zoomed.write(args.output_emission+f"_subject_{args.subject}.hv")
    mu_zoomed.write(args.output_mu+f"_subject_{args.subject}.hv")

    # --- 9. Optional Plotting ---
    if args.save_plot:
        import matplotlib.pyplot as plt
        print(f"Generating plot and saving to {args.save_plot}")
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        emission_arr = emission_zoomed.as_array()
        mu_arr = mu_zoomed.as_array()
        
        z_slice = emission_arr.shape[0] // 2
        
        im0 = ax[0].imshow(emission_arr[z_slice, :, :], cmap='viridis')
        ax[0].set_title(f'Zoomed Emission (FDG) - Slice {z_slice}')
        fig.colorbar(im0, ax=ax[0])
        
        im1 = ax[1].imshow(mu_arr[z_slice, :, :], cmap='gray')
        ax[1].set_title(f'Zoomed Mu-map - Slice {z_slice}')
        fig.colorbar(im1, ax=ax[1])
        
        plt.tight_layout()
        plt.savefig(args.save_plot)

    print("Script finished successfully.")


if __name__ == "__main__":
    main()