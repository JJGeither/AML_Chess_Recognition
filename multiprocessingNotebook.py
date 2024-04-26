from skimage import color, exposure
from skimage.feature import hog
from skimage.io import imread
from skimage.util import view_as_blocks
import numpy as np
import multiprocessing as mp
import os
import sys


def runt(fileName, src, fen_from_filename, fen_from_position, NUM_SPACES_PER_BOARD, data_dict):
    
    def HogTransform(img):
        first_image_gray = color.rgb2gray(img)

        fd, hog_image = hog(
            first_image_gray,
            orientations=8,
            pixels_per_cell=(8, 8),
            cells_per_block=(1, 1),
            visualize=True,
            block_norm='L2-Hys',
            feature_vector=True
        )

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_image_uint8 = (hog_image_rescaled * 255).astype(np.uint8)
        return hog_image_uint8


    data = {'fenstring': [], 'data': []}

    dimension = 400 // 8
    spaceTilesStored, count = 0, 0

    if fileName.endswith('.jpg') or fileName.endswith('.jpeg'):
        im = imread(os.path.join(src, fileName))
        fenString = fen_from_filename(fileName)

        patches = view_as_blocks(im, block_shape=(dimension, dimension, 3)).reshape(-1, dimension, dimension, 3)
        boardPosition = 0
        
        for patch in patches:
            patchType = fen_from_position(boardPosition, fenString)
            if patchType == ' ':
                if spaceTilesStored > NUM_SPACES_PER_BOARD:
                    continue
                spaceTilesStored += 1
            
            hog_features = HogTransform(patch)
            #hog_features = patch

            # Append to shared lists in the manager dict
            # Assuming data_dict is a list of dictionaries

            #with lock:
            new_data = {'data': [hog_features], 'fenstring': [patchType]}
            data_dict.append(new_data)

            boardPosition += 1
    
    return
