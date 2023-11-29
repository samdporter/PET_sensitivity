import os
import sys
import argparse
import torch
from pathlib import Path

from sirf.STIR import MessageRedirector, ImageData, AcquisitionData, AcquisitionModelUsingRayTracingMatrix
import sirf.STIR
from sirf.Utilities import examples_data_path

dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
source_path = os.path.join(dir_path, 'source')
sys.path.append(source_path)

from data.ellipses import EllipsesDataset

def parse_arguments():
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description='PET Sensitivity Data Preparation')
    parser.add_argument('-n', '--num_samples', type=int, default=2**16, help='Number of samples')
    parser.add_argument('-s', '--save_path', type=str, default=f'{dir_path}/data/training_data', help='Path to save data')
    parser.add_argument('-d', '--data_path', type=str, default=f'{dir_path}/data/template_data', help='Path to data')
    parser.add_argument('-f', '--filename', type=str, default='ellipses', help='Filename of saved data')
    parser.add_argument('-g', '--generate_non_attenuated_sensitivity', action='store_true', help='Generate non-attenuated sensitivity')
    return parser.parse_args()

def setup_device():
    """
    Setup the device for PyTorch.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args=None):

    # Redirect messages and set verbosity
    msg = MessageRedirector()
    sirf.STIR.set_verbosity(0)

    attn_image = ImageData(os.path.join(args.data_path, 'attenuation.hv'))
    template = AcquisitionData(os.path.join(args.data_path, 'template_sinogram.hs'))

    train_dataloader = torch.utils.data.DataLoader(
        EllipsesDataset(attn_image, template,  num_samples=args.num_samples, generate_non_attenuated_sensitivity=args.generate_non_attenuated_sensitivity),
        batch_size=args.num_samples, shuffle=True)

    # Data processing
    X_train, y_train = process_data(train_dataloader)

    # Save data to file
    save_data(args, X_train, y_train)

    # Remove temporary files
    remove_temporary_files()

def process_data(dataloader):
    """
    Process data using the provided dataloader.
    """
    X_train = []
    y_train = []
    for train in dataloader:
        X, y = train
        X_train.append(X)
        y_train.append(y)
    return torch.cat(X_train, dim=0), torch.cat(y_train, dim=0)

def save_data(args, X_train, y_train):
    """
    Save data to file.
    """
    if args.generate_non_attenuated_sensitivity:
        save_path = os.path.join(args.save_path, 'plus_non_attenuated')
    else:
        save_path = os.path.join(args.save_path, 'original')

    x_filename = f'{args.filename}_X_train_n{args.num_samples}_0.pt'
    y_filename = f'{args.filename}_y_train_n{args.num_samples}_0.pt'

    x_file = os.path.join(save_path, x_filename)
    y_file = os.path.join(save_path, y_filename)

    if Path(x_file).exists():
        i = 1
        while (Path(os.path.join(save_path, f'{args.filename}_X_train_n{args.num_samples}_{i}.pt'))).exists():
            i += 1
        x_file = os.path.join(save_path, f'{args.filename}_X_train_n{args.num_samples}_{i}.pt')
        y_file = os.path.join(save_path, f'{args.filename}_y_train_n{args.num_samples}_{i}.pt')

    torch.save(X_train, x_file)
    torch.save(y_train, y_file)

def remove_temporary_files():
    """
    Remove temporary files.
    """

    for filename in os.listdir(dir_path):
        if filename.startswith('tmp_') and (filename.endswith('.s') or filename.endswith('.hs')):
            os.remove(filename)
   
if __name__ == '__main__':
    main(parse_arguments())