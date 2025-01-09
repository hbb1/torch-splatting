import math
import time

import torch
import numpy as np
from plyfile import PlyData
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer
from gaussian_splatting.utils.camera_utils import Camera, to_viewpoint_camera
import gaussian_splatting.utils as utils


def load_ply_model(path):
    """Load a PLY file and create a GaussModel from it."""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']

    # Extract xyz coordinates
    xyz = np.column_stack((vertices['x'], vertices['y'], vertices['z']))

    # Extract features (both DC and rest)
    feature_names = [prop.name for prop in vertices.properties if prop.name.startswith('f_')]
    dc_names = [name for name in feature_names if 'dc' in name]
    rest_names = [name for name in feature_names if 'rest' in name]

    f_dc = np.column_stack([vertices[name] for name in dc_names])
    f_rest = np.column_stack([vertices[name] for name in rest_names])

    # Extract scaling parameters
    scale_names = [prop.name for prop in vertices.properties if prop.name.startswith('scale_')]
    scales = np.column_stack([vertices[name] for name in scale_names])

    # Extract rotation parameters
    rot_names = [prop.name for prop in vertices.properties if prop.name.startswith('rot_')]
    rotations = np.column_stack([vertices[name] for name in rot_names])

    # Extract opacity
    opacity = vertices['opacity'].reshape(-1, 1)

    # Calculate dimensions for spherical harmonics
    # num_bands = sh_degree + 1
    # num_coeffs_rest = num_bands * num_bands - 1  # subtract 1 for DC term
    num_coeffs_rest = int(f_rest.shape[1] / 3)
    sh_degree_f = np.sqrt(num_coeffs_rest + 1) - 1
    assert sh_degree_f.is_integer()
    sh_degree = int(sh_degree_f)

    # Create and initialize GaussModel
    model = GaussModel(sh_degree=sh_degree)

    model._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float32).cuda())
    model._features_dc = torch.nn.Parameter(torch.tensor(f_dc, dtype=torch.float32).reshape(-1, 1, 3).cuda())
    model._features_rest = torch.nn.Parameter(
        torch.tensor(f_rest, dtype=torch.float32).reshape(-1, 3, num_coeffs_rest).transpose(2, 1).cuda())

    model._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float32).cuda())
    model._rotation = torch.nn.Parameter(torch.tensor(rotations, dtype=torch.float32).cuda())
    model._opacity = torch.nn.Parameter(torch.tensor(opacity, dtype=torch.float32).cuda())

    return model


def create_camera_from_params(intrinsic_matrix, pose_matrix, width, height):
    """Create camera from provided intrinsic and pose parameters."""
    intrinsic = np.eye(4)
    intrinsic[:3, :3] = np.array(intrinsic_matrix)

    c2w = np.array(pose_matrix)

    return torch.tensor(intrinsic, dtype=torch.float32).cuda(), torch.tensor(c2w, dtype=torch.float32).cuda()


if __name__ == '__main__':
    ply_path = 'result/test/splats-200.ply'
    output_path = 'result/test/image2-200b.png'

    # Load the model
    print(f"Loading PLY file from {ply_path}")
    model = load_ply_model(ply_path)

    W = 512
    H = 512
    intrinsic = torch.tensor([
        [711.1110599640117, 0, 256, 0],
        [0, 711.1110599640117, 256, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]).cuda()
    assert intrinsic[0, 2] == W/2, "change scale instead"

    scale = .5
    intrinsic[:2, :4]*=scale
    W*=scale
    H*=scale

    fov_h = math.degrees(2 * math.atan2(intrinsic[0, 2], intrinsic[0, 0]))
    print(f"intrinsic {int(W)}x{int(H)} - camera {fov_h:.1f}°")

    # colmap orientation (see data_utils.py read_camera)
    c2w = torch.tensor([
        [-.86086243, .37950450, -.33895749, .67791492],
        [.50883776, .64205378, -.57345545, 1.1469108],
        [1.0933868e-08, -.66614062, -.74582618, 1.4916525],
        [0, 0, 0, 1]
    ]).cuda()

    camera = Camera(width=W, height=H, intrinsic=intrinsic, c2w=c2w)

    renderer = GaussRenderer(
        active_sh_degree=model.max_sh_degree,
        white_bkgd=True
    )

    # Render
    start_time = time.time()
    with torch.no_grad():
        out = renderer(pc=model, camera=camera)
    print(f"Rendering took {time.time() - start_time:.3f} seconds")

    # Save the rendered image
    image = out['render'].detach().cpu().numpy()
    utils.imwrite(output_path, image)
    print(f"Saved rendered image to {output_path}")
