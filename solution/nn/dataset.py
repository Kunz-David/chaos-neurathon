from functools import lru_cache
from multiprocessing import Pool
import time

import yaml
from PIL import Image

from nn.util import *

import logging

logger = logging.getLogger()


def load_img(text):
    img_path, mode = text.split("#")
    return np.array(Image.open(img_path).convert(mode))


def load_img_path(text):
    img_path, mode = text.split("#")
    return img_path, np.array(Image.open(img_path).convert(mode))


# Use a custom dataset to read in albedo <--> (bump, roughness) map pairs
class CustomDataset(Dataset):
    def __init__(self, config, albedo_dirs, bump_dirs, rough_dirs, transform=None):
        self.transform = transform
        self.config = config

        self.allowed_albedo_filenames = []
        with open(config['clusters_filepath'], 'r') as f:
            clusters = yaml.load(f, Loader=yaml.FullLoader)
        for cluster in config['clusters']:
            self.allowed_albedo_filenames += [os.path.split(filepath)[-1] for filepath in clusters[cluster]]

        self.train_with_patches = config['train_with_patches']
        self.patch_size = config['patch_size']
        self.num_patches = config['num_patches']

        # Initialize lists to store file paths
        self.albedo_paths = []
        self.bump_paths = []
        self.rough_paths = []

        # Iterate over each set of directories and collect file paths
        for albedo_dir, bump_dir, rough_dir in zip(albedo_dirs, bump_dirs, rough_dirs):
            albedo_files = [f for f in os.listdir(albedo_dir) if is_albedo_img(f)]

            for file in albedo_files:
                if file not in self.allowed_albedo_filenames and len(self.allowed_albedo_filenames) > 0:
                    continue

                common_prefix = albedo_basename(file)

                albedo_file = os.path.join(albedo_dir, file)
                if not os.path.exists(albedo_file):
                    raise Exception("Albedo file {} doesn't exist.".format(albedo_file))

                self.albedo_paths.append(albedo_file)

                bump_file = os.path.join(bump_dir, bump_filename(common_prefix))
                if not os.path.exists(bump_file):
                    raise Exception("Bump map file {} doesn't exist.".format(bump_file))

                self.bump_paths.append(bump_file)

                rough_file = os.path.join(rough_dir, roughness_filename(common_prefix))
                if not os.path.exists(rough_file):
                    # Try look for differently named file with a capital 'Raw'
                    rough_file = os.path.join(rough_dir, roughness_filename(common_prefix, fourk=True))
                    if not os.path.exists(rough_file):
                        # Try look for differently named file with a capital 'Raw'
                        rough_file = os.path.join(rough_dir, roughness_filename(common_prefix, capital_raw=True))
                        if not os.path.exists(rough_file):
                            # Try look for differently named file with 'Mask' in the filename
                            rough_file = os.path.join(rough_dir, roughness_filename(common_prefix, mask=True))
                            if not os.path.exists(rough_file):
                                # Try look for differently named file with '_Roughness' in the filename
                                rough_file = os.path.join(rough_dir, roughness_filename(common_prefix, cos=True))
                                if not os.path.exists(rough_file):
                                    # Try look for differently named file with '_Roughness (1)' in the filename
                                    rough_file = os.path.join(rough_dir, roughness_filename(common_prefix, cos=True, copy=True))
                                    if not os.path.exists(rough_file):
                                        raise Exception("Roughness map file {} doesn't exist (albedo {}).".format(rough_file, file))

                self.rough_paths.append(rough_file)

        logger.info('There is %d albedo images, %d bump images, %d roughness images' % (len(self.albedo_paths), len(self.bump_paths), len(self.rough_paths)))

        pool = Pool(12)
        logger.info('Loading albedo images...')
        start = time.time()
        self.albedo_images = pool.map(load_img, [path + '#RGB' for path in self.albedo_paths])
        logger.info('Loading albedo images took %.1f s...', time.time() - start)

        start = time.time()
        logger.info('Loading bump images...')
        self.bump_images = pool.map(load_img, [path + '#L' for path in self.bump_paths])
        logger.info('Loading bump images took %.1f s...', time.time() - start)

        start = time.time()
        logger.info('Loading rough images...')
        self.rough_images = pool.map(load_img, [path + '#L' for path in self.rough_paths])
        logger.info('Loading rough images took %.1f s...', time.time() - start)

        pool.close()

        assert len(self.albedo_paths) == len(self.bump_paths) == len(self.rough_paths), "Mismatch in dataset sizes"

    def cut_patches(self, patch_size, images):
        img_height, img_width, c = images[0].shape

        if self.config['tile']:
            # Create a 2x2 tiled version of each image
            images = [self.tile_image(image) for image in images]

        # Randomly choose a top-left point for the patch in the original image dimensions
        y = np.random.randint(0, img_height-patch_size)
        x = np.random.randint(0, img_width-patch_size)

        # Compute patch coordinates
        x1, y1 = x, y
        x2, y2 = x + patch_size, y + patch_size

        # Crop the patch from the tiled image
        images = [image[y1:y2, x1:x2] for image in images]

        return images

    def tile_image(self, image):
        tiled_image = np.concatenate((image, image), axis=1)
        tiled_image = np.concatenate((tiled_image, tiled_image), axis=0)
        return tiled_image

    def __getitem__(self, index):
        img_index = index // self.num_patches

        # Load images
        image_albedo = self.albedo_images[img_index]
        image_bump = self.bump_images[img_index]
        image_rough = self.rough_images[img_index]

        images = [image_albedo, image_bump, image_rough]

        if self.train_with_patches:
            images = self.cut_patches(self.patch_size, images)

        if self.transform:
            images = [self.transform(image) * 2 - 1 for image in images]

        return img_index, images

    def __len__(self):
        return len(self.albedo_paths) * self.num_patches


class AlbedoDataset(Dataset):
    def __init__(self, albedo_dirs, transform=None):
        """
        albedo_dirs: List of directories containing albedo images.
        """
        self.transform = transform

        # Initialize a list to store file paths
        self.albedo_images = []

        jobs = []
        # Iterate over each directory and collect file paths
        for albedo_dir in albedo_dirs:
            albedo_files = os.listdir(albedo_dir)
            for file in albedo_files:
                albedo_filepath = os.path.join(albedo_dir, file)
                if os.path.isdir(albedo_filepath):
                    continue
                jobs.append(albedo_filepath)

        pool = Pool(12)
        results = pool.map(load_img_path, [path + '#RGB' for path in jobs])
        self.albedo_image_paths = [result[0] for result in results]
        self.albedo_images = [result[1] for result in results]

        pool.close()

    def __len__(self):
        return len(self.albedo_images)

    def __getitem__(self, idx):
        albedo_image = self.albedo_images[idx]

        if self.transform:
            albedo_image = self.transform(albedo_image) * 2 - 1

        return albedo_image

