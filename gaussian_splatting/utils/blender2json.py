"""
Script to parse the generated files into our desirable format
"""
import argparse
import json
import math
import os
import random
import sys
import glob
from PIL import Image
import numpy as np
import pdb

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    args = parser.parse_args()
    
    input_path = args.input_path
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    info = json.load(open(os.path.join(input_path, 'info.json')))
    
    object = dict(images=[])
    num_images = len(glob.glob(f'{input_path}/*[0-9].json'))
    for i in range(num_images):
        camera_metadata = json.load(open(os.path.join(input_path, f"{i:05}.json")))
        channels = [np.array(Image.open(os.path.join(input_path, f"{i:05}_{c}.png"))) / (2**16-1) for c in ['r', 'g', 'b', 'a', 'depth']]
        
        # mapping to [0,255]
        rgb = np.uint8(np.stack(channels[:3], axis=-1) * 255)
        mask = np.uint8(channels[3] * 255)
        depth = np.uint8(channels[-1] * 255)

        # merge image
        Image.fromarray(rgb).convert('RGB').save(os.path.join(output_path, f'{i:05}_rgb.png'))
        Image.fromarray(depth).save(os.path.join(output_path, f'{i:05}_depth.png'))
        Image.fromarray(mask).save(os.path.join(output_path, f'{i:05}_alpha.png'))

        # intrinsic
        x_fov, y_fov = camera_metadata['x_fov'], camera_metadata['y_fov']
        width, height = rgb.shape[:2]
        fx = 1 / np.tan(x_fov / 2) * (width / 2)
        fy = 1 / np.tan(y_fov / 2) * (height / 2)
        intrinsic = np.array([
            [fx, 0, width/2],
            [0, fy, height/2],
            [0, 0, 1]
        ])

        """
            OpenGL (X right; Y up; Z inward) 
        """

        # write extrinsic from opencv camera
        origin = np.array(camera_metadata['origin']).reshape(-1,1) # origin
        right = np.array(camera_metadata['x']).reshape(-1,1)  # right vector
        down = np.array(camera_metadata['y']).reshape(-1,1)   # down vector
        lookat = np.array(camera_metadata['z']).reshape(-1,1) # forward vector

        pose = np.block([
                [right, -down,  -lookat,  origin],
                [0., 0., 0., 1.]]
            )

        image = dict(
            intrinsic= intrinsic.tolist(),
            pose = pose.tolist(),
            rgb=os.path.join(f'{i:05}_rgb.png'),
            depth=os.path.join(f'{i:05}_depth.png'),
            alpha=os.path.join(f'{i:05}_alpha.png'),
            max_depth = camera_metadata['max_depth'],
            HW = [height, width],
        )

        object['images'].append(image)
        object['bbox'] = camera_metadata['bbox'] 

    with open(os.path.join(output_path, 'info.json'), "w") as f:
        object = {**info, **object}
        json.dump(object, f)

main()