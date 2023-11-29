# This is functionality extracted from [Operator Discretization Library (ODL)]
# (https://odlgroup.github.io/odl/index.html) and changed somewhat for our needs.
# The appropriate pieces of code that are used are: [here]
# (https://github.com/odlgroup/odl/blob/master/odl/phantom/transmission.py)
# and [here](https://github.com/odlgroup/odl/blob/master/odl/phantom/geometric.py).

# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
from scipy.ndimage import affine_transform

import random
import os
import brainweb


def _getshapes_2d(center, max_radius, shape):
    """Calculate indices and slices for the bounding box of a disk."""
    index_mean = shape * center
    index_radius = max_radius / 2.0 * np.array(shape)

    # Avoid negative indices
    min_idx = np.maximum(np.floor(index_mean - index_radius), 0).astype(int)
    max_idx = np.ceil(index_mean + index_radius).astype(int)
    idx = [slice(minx, maxx) for minx, maxx in zip(min_idx, max_idx)]
    shapes = [(idx[0], slice(None)),
              (slice(None), idx[1])]
    return tuple(idx), tuple(shapes)

def ellipse_phantom(shape, ellipses):
    
    """Create a phantom of ellipses in 2d space.

    Parameters
    ----------
    shape : `tuple`
        Size of image
    ellipses : list of lists
        Each row should contain the entries ::

            'value',
            'axis_1', 'axis_2',
            'center_x', 'center_y',
            'rotation'

        The provided ellipses need to be specified relative to the
        reference rectangle ``[-1, -1] x [1, 1]``. Angles are to be given
        in radians.

    Returns
    -------
    phantom : numpy.ndarray
        2D ellipse phantom.

    See Also
    --------
    shepp_logan : The typical use-case for this function.
    """
    # Blank image
    p = np.zeros(shape)

    grid_in = (np.expand_dims(np.linspace(0, 1, shape[0]),1),
                    np.expand_dims(np.linspace(0, 1, shape[1]),0))

    # move points to [-1, 1]
    grid = []
    for i in range(2):
        mean_i = 0.5
        # Where space.shape = 1, we have minp = maxp, so we set diff_i = 1
        # to avoid division by zero. Effectively, this allows constructing
        # a slice of a 2D phantom.
        diff_i = 0.5
        grid.append((grid_in[i] - mean_i) / diff_i)

    for ellip in ellipses:
        assert len(ellip) == 6

        intensity = ellip[0]
        a_squared = ellip[1] ** 2
        b_squared = ellip[2] ** 2
        x0 = ellip[3]
        y0 = ellip[4]
        theta = ellip[5]

        scales = [1 / a_squared, 1 / b_squared]
        center = (np.array([x0, y0]) + 1.0) / 2.0

        # Create the offset x,y and z values for the grid
        if theta != 0:
            # Rotate the points to the expected coordinate system.
            ctheta = np.cos(theta)
            stheta = np.sin(theta)

            mat = np.array([[ctheta, stheta],
                            [-stheta, ctheta]])

            # Calculate the points that could possibly be inside the volume
            # Since the points are rotated, we cannot do anything directional
            # without more logic
            max_radius = np.sqrt(
                np.abs(mat).dot([a_squared, b_squared]))
            idx, shapes = _getshapes_2d(center, max_radius, shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            offset_points = [vec * (xi - x0i)[..., None]
                             for xi, vec, x0i in zip(subgrid,
                                                     mat.T,
                                                     [x0, y0])]
            rotated = offset_points[0] + offset_points[1]
            np.square(rotated, out=rotated)
            radius = np.dot(rotated, scales)
        else:
            # Calculate the points that could possibly be inside the volume
            max_radius = np.sqrt([a_squared, b_squared])
            idx, shapes = _getshapes_2d(center, max_radius, shape)

            subgrid = [g[idi] for g, idi in zip(grid, shapes)]
            squared_dist = [ai * (xi - x0i) ** 2
                            for xi, ai, x0i in zip(subgrid,
                                                   scales,
                                                   [x0, y0])]

            # Parentheses to get best order for broadcasting
            radius = squared_dist[0] + squared_dist[1]

        # Find the points within the ellipse
        inside = radius <= 1

        # Add the ellipse intensity to those points
        p[idx][inside] += intensity
    return p

def random_2D_rotation():
    """ Generate a random 2D rotation matrix in 3D space"""
    phi = np.random.uniform(-0.05, 0.05)
    c = np.cos(phi)
    s = np.sin(phi)
    R = np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])
    return R
    
def random_4x4_matrix():
    """Generate a random 4x4 matrix"""
    A = np.eye(4)
    Sy, Sz = np.random.random()*0.5+0.75, np.random.random()*0.5+0.75
    A[1,1], A[2,2] = Sy, Sz
    Rx = random_2D_rotation()
    return np.dot(Rx, A)

def affine_transform_2D(theta, tx, ty, sx, sy, image_arr):
    ''' create a random affine transformation for 2D images '''
    # create the transformation matrix
    transformation_matrix = np.array([[sx*np.cos(theta), -sy*np.sin(theta), tx],
                                        [sx*np.sin(theta),  sy*np.cos(theta), ty],
                                        [0, 0, 1]])

    # apply the transformation
    image_arr_transformed = affine_transform(image_arr, transformation_matrix, order=1)
    return image_arr_transformed

def affine_transform_2D_image(theta, tx, ty, sx, sy, image):
    ''' create a random affine transformation for 2D images '''
    res = image.clone()
    res_arr = affine_transform_2D(theta, tx, ty, sx, sy, np.squeeze(image.as_array()))
    res.fill(np.expand_dims(res_arr, 0))
    return res

def random_shapes():
    x_0 = 1 * np.random.rand() - 0.5
    y_0 = 1 * np.random.rand() - 0.5
    return [np.random.exponential(0.4),
            1 * np.random.rand() - 0.5, 1 * np.random.rand() - 0.5,
            x_0, y_0,
            np.random.rand() * 2 * np.pi]

def random_phantom(space, n_ellipse=20):
    n = np.random.poisson(n_ellipse)
    shapes = [random_shapes() for _ in range(n)]
    for i in range(n):
        shapes[i][0] = np.random.exponential(0.4)*np.random.random()
    x = ellipse_phantom(space[1:], shapes)
    x = [x]
    return np.array(x)

def shepp_logan(space):
    rad18 = np.deg2rad(18.0)
    #            value  axisx  axisy     x       y  rotation
    ellipsoids= [[0.55, 0.69, 0.92, 0.0, 0.0, 0],
                [0.60, 0.6624, 0.874, 0.0, -0.0184, 0],
                [0.50, 0.11, 0.31, 0.22, 0.0, -rad18],
                [0.51, 0.16, 0.41, -0.22, 0.0, rad18],
                [0.05, 0.21, 0.25, 0.0, 0.35, 0],
                [0.11, 0.046, 0.046, 0.0, 0.1, 0],
                [0.48, 0.046, 0.046, 0.0, -0.1, 0],
                [0.34, 0.046, 0.023, -0.08, -0.605, 0],
                [0.14, 0.023, 0.023, 0.0, -0.606, 0],
                [1.28, 0.023, 0.046, 0.06, -0.605, 0]]
    x = ellipse_phantom(space[1:], ellipsoids)
    x = [x]
    return np.array(x)

def random_camouflage_array(shape, num_patches, min_size=2, max_size=10, colors=[0, 128, 255]):
    # Initialize an empty array
    canvas = np.zeros(shape)
    
    for i in range(num_patches):
        # Generate random parameters for the patch
        center_x = np.random.randint(low=0, high=shape[1])
        center_y = np.random.randint(low=0, high=shape[0])
        size = np.random.randint(low=min_size, high=max_size)
        color = random.choice(colors)
        
        # Compute the patch boundaries
        x1 = center_x - size // 2
        y1 = center_y - size // 2
        x2 = center_x + size // 2
        y2 = center_y + size // 2
        
        # Clip the patch boundaries to the canvas boundaries
        x1 = np.clip(x1, 0, shape[1]-1)
        y1 = np.clip(y1, 0, shape[0]-1)
        x2 = np.clip(x2, 0, shape[1]-1)
        y2 = np.clip(y2, 0, shape[0]-1)
        
        # Fill the patch with the color
        canvas[y1:y2, x1:x2] = color
    
    return canvas

def random_polygons_array(shape, num_polygons, min_n=3, max_n=8, min_size=5, max_size=50):
    
    canvas = np.zeros(shape)

    for i in range(num_polygons):
        # Generate random parameters for the polygon
        center_x = np.random.randint(low=0, high=shape[1])
        center_y = np.random.randint(low=0, high=shape[0])
        size = np.random.randint(low=min_size, high=max_size)
        n = np.random.randint(low=min_n, high=max_n)
        color = np.random.randint(low=1, high=255)
        
        # Generate the vertices of the polygon
        theta = np.linspace(0, 2*np.pi, n, endpoint=False)
        x = center_x + size*np.cos(theta)
        y = center_y + size*np.sin(theta)
        
        # Convert the vertices to integer coordinates
        x = np.round(x).astype(int)
        y = np.round(y).astype(int)
        
        # Clip the vertices to the canvas boundaries
        x = np.clip(x, 0, shape[1]-1)
        y = np.clip(y, 0, shape[0]-1)
        
        # Fill the polygon with the color
        for j in range(n):
            x1, y1 = x[j], y[j]
            x2, y2 = x[(j+1) % n], y[(j+1) % n]
            canvas = fill_line(canvas, x1, y1, x2, y2, color)
    
    return canvas

def fill_line(canvas, x1, y1, x2, y2, color):
    # Compute the line segment parameters
    dx = x2 - x1
    dy = y2 - y1
    d = max(abs(dx), abs(dy))
    # Avoid division by zero
    if d == 0:
        return canvas

    sx = dx / d
    sy = dy / d
    
    # Iterate over the line segment pixels
    x = x1
    y = y1
    for i in range(d+1):
        canvas[int(y), int(x)] = color
        x += sx
        y += sy
    
    return canvas

def random_shapes_array(shape, num_shapes, min_size=2, max_size=20):

    canvas = np.zeros(shape)
    
    for _ in range(num_shapes):
        # Generate random parameters for the shape
        center_x = np.random.randint(low=0, high=shape[1])
        center_y = np.random.randint(low=0, high=shape[0])
        size = np.random.randint(low=min_size, high=max_size)
        color = np.random.randint(low=1, high=255)
        
        # Choose a random shape
        choose_shape = np.random.choice(['rectangle', 'circle'])
        
        if choose_shape == 'rectangle':
            x_min = center_x - size//2
            x_max = center_x + size//2
            y_min = center_y - size//2
            y_max = center_y + size//2
            
            # Clip the rectangle to the canvas boundaries
            x_min = max(0, x_min)
            x_max = min(shape[1], x_max)
            y_min = max(0, y_min)
            y_max = min(shape[0], y_max)
            
            # Fill the rectangle with the color
            canvas[y_min:y_max, x_min:x_max] = color
        
        elif choose_shape == 'circle':

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

            # Compute the distances from the center of the circle
            distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            
            # Select the points within the circle
            mask = distances <= size//2
            
            # Fill the circle with the color
            canvas[mask] = color
    
    return canvas

def crop_to_circle(arr, r):
    """ 
    Crop 2D array to circle of radius r
    with centre at centre of array
    return array with zeros outside circle
    """
    x, y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
    distances = np.sqrt((x - arr.shape[1]//2)**2 + (y - arr.shape[0]//2)**2)
    mask = distances <= r
    return arr * mask

def get_brainweb_images(n_files=20, brainweb_filepath = "/home/sam/working/SIRF-Exercises/notebooks/PET_sensitivity/brainweb_subjects"):

    # Check if filepath exists and whether it contains n_files files. If not, download them
    if not os.path.exists(brainweb_filepath) and len(os.listdir(brainweb_filepath)) != n_files:
        print("Downloading brainweb files...")
        files, data, fnames = [], [], []
        for i in range(n_files):
            fname, url= sorted(brainweb.utils.LINKS.items())[i]
            file = brainweb.get_file(fname, url, ".")
            datum = brainweb.load_file(fname)
            files.append(file)
            data.append(datum)
            fnames.append(os.path.join(brainweb_filepath, file))
    else:
        print("Loading brainweb files...")
        file_paths = sorted(os.listdir(brainweb_filepath))
        files, data, fnames, i = [], [], [], 0
        for file in file_paths:
            if file.endswith(".gz") and i < n_files:
                datum = brainweb.load_file(os.path.join(brainweb_filepath, file))
                data.append(datum)
                fnames.append(os.path.join(brainweb_filepath, file))
                i += 1