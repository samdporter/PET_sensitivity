import numpy as np
import matplotlib.pyplot as plt
import imageio


def plot_pet_acquisition(image, sinogram, angles, output_filename):
    frames = []  # To store the frames for the GIF
    
    # Create a figure for plotting
    fig, (ax_image, ax_sinogram) = plt.subplots(1, 2, figsize=(10, 5))
    
    for i, angle in enumerate(angles):
        # Clear previous images
        ax_image.clear()
        ax_sinogram.clear()
        
        # Plot PET image
        ax_image.imshow(image, cmap='viridis')
        ax_image.set_title('PET Image')
        ax_image.axis('off')

        # Draw the projection lines
        r = image.shape[0] / 2
        origin = (image.shape[1] // 2, image.shape[0] // 2)   
        pos_x = r * np.cos(np.deg2rad(angle)+np.pi/2) + origin[0]
        pos_y = r * np.sin(np.deg2rad(angle)+np.pi/2) + origin[1]
        neg_x = -r * np.cos(np.deg2rad(angle)+np.pi/2) + origin[0]
        neg_y = -r * np.sin(np.deg2rad(angle)+np.pi/2) + origin[1]
        x = [pos_x, neg_x]
        y = [pos_y, neg_y]
        # draw the line with an arrow at the end
        ax_image.plot(x, y, 'r-')
        # print angle to 1 decimal place
        ax_image.text(20, 10, f'{np.round(angle, 1)}°', color='r', fontsize=16, ha='center', va='center')

        # Plot Sinogram
        ax_sinogram.imshow(sinogram, cmap='viridis', aspect='auto')
        ax_sinogram.set_title('Sinogram')
        ax_sinogram.axhline(y=i, color='r', linestyle='--')
        # print angle on sinogram
        ax_sinogram.text(20, i, f'{np.round(angle, 1)}°', color='r', fontsize=16, ha='center', va='center')

        # Draw the frames
        fig.canvas.draw()
        
        # Convert the plot to image for the GIF
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image_from_plot)

    # Save the frames as a GIF
    imageio.mimsave(output_filename, frames, fps=5)

    # prevent the figure from being displayed
    plt.close()