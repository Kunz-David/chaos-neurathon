import io
import logging
import os
import pathlib
import shutil
import glob
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
import torchvision.transforms.functional as TF

import numpy as np

from PIL import Image
from tqdm import tqdm
import zipfile

import matplotlib.pyplot as plt
from IPython.display import clear_output

# Collection of various utility functions
def is_albedo_img(f):
    return '_Diff_4k_srgb' in f or '_Diff_2k_srgb' in f or '_Color' in f


# Extracting the common basename of a file for an
# albedo
def albedo_basename(f):
    if '_Diff_4k_srgb' in f:
        return f.split('_Diff_4k_srgb')[0]
    elif '_Diff_2k_srgb' in f:
        return f.split('_Diff_2k_srgb')[0]
    elif '_Color' in f:
        return f.split('_Color')[0]
    return None


def bump_filename(common_prefix):
    return common_prefix + '_bump.png'


# Handling the various conventions associated with roughness map naming
def roughness_filename(common_prefix, capital_raw=False, mask=False, cos=False, copy=False, fourk=False):
    if copy:
        copy_suffix = ' (1)'
    else:
        copy_suffix = ''

    if cos:
        return common_prefix + '_Roughness' + copy_suffix + '.png'

    if mask:
        mask_text = 'Mask_'
    else:
        mask_text = ''
    if capital_raw:
        raw_text = 'Raw'
    else:
        raw_text = 'raw'

    if fourk:
        return common_prefix + '_Rough_' + mask_text + '2k_' + raw_text + copy_suffix + '.png'

    return common_prefix + '_Rough_' + mask_text + '4k_' + raw_text + copy_suffix + '.png'


# Function to update the loss plot
def live_plot(data_dict, figsize=(7, 5), title='', x_label='epoch', log_losses=False):
    clear_output(wait=True)
    plt.figure(figsize=figsize)
    for label, data in data_dict.items():
        if log_losses:
            plt.plot(np.log(data), label=label)
        else:
            plt.plot(data, label=label)
    plt.title(title)
    plt.grid(True)
    plt.xlabel(x_label)
    if log_losses:
        plt.ylabel('log(loss)')
    else:
        plt.ylabel('loss')
    plt.legend(loc='center left')  # Change if needed
    plt.show()


# Function to save predictions
def save_prediction(output, output_filename, output_dir):

    # Post-process the output
    output_image = output.squeeze(0).cpu()  # Remove batch dimension and move to cpu
    output_image = TF.to_pil_image(output_image / 2 + 0.5)  # Convert to PIL image

    # NOTE: Saved bump and roughness images should be 1 channel greyscale images

    # Construct output file path
    output_path = os.path.join(output_dir, output_filename)

    # Save the output image
    output_image.save(output_path)


class TqdmToLogger(io.StringIO):
    """
        Output stream for TQDM which will output to logger module instead of
        the StdOut.
    """
    logger = None
    level = None
    buf = ''

    def __init__(self, logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level or logging.INFO

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

    def flush(self):
        self.logger.log(self.level, self.buf)
