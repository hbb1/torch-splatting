import torch
import random
from typing import BinaryIO, Dict, List, Optional, Union
import numpy as np
from  gaussian_splatting.utils.camera_utils import parse_camera


def get_rays_single_image(H, W, intrinsics, c2w, render_stride=1):
        """
        :param H: image height H
        :param W: image width W
        :param intrinsics: 4 by 4 intrinsic matrix [B, 4, 4]
        :param c2w: 4 by 4 camera to world extrinsic matrix [B, 4, 4]
        :return: rays_o, rays_d [B, HxW, 3] [B, HxW, 3]
        """
        device = intrinsics.device
        # indexing = x, y
        u, v = np.meshgrid(np.arange(W)[:: render_stride], np.arange(H)[:: render_stride])
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels).to(device)
        batched_pixels = pixels.unsqueeze(0).repeat(len(intrinsics), 1, 1)
        rays_d = (
            c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)
        ).transpose(1, 2)

        rays_o = (
            c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[1], 1)
        )  # B x HW x 3

        return rays_o, rays_d


def get_point_clouds(cameras, depths, alphas, rgbs=None):
    """
    depth map to point cloud
    """
    Hs, Ws, intrinsics, c2ws = parse_camera(cameras)
    W, H = int(Ws[0].item()), int(Hs[0].item())
    assert (depths.shape == alphas.shape)
    coords = []
    rgbas = []
    rays_o, rays_d = get_rays_single_image(H=H, W=W, intrinsics=intrinsics, c2w=c2ws)
    mask = (alphas.flatten(1) == 1)
    pts = rays_o + rays_d * depths.flatten(1).unsqueeze(-1)
    rgbas = torch.cat([rgbs, alphas.unsqueeze(-1)], dim=-1)
    coords = pts[mask].cpu().numpy()
    rgbas = rgbas.flatten(1,-2)[mask].cpu().numpy()

    if rgbs is not None:
        channels = dict(
            R=rgbas[..., 0],
            G=rgbas[..., 1],
            B=rgbas[..., 2],
            A=rgbas[..., 3],
        )
    else:
        channels = {}

    point_cloud = PointCloud(coords, channels)
    return point_cloud


"""
    adpoted from https://github.com/openai/point-e/blob/main/point_e/util/point_cloud.py
"""
COLORS = frozenset(["R", "G", "B", "A"])

def preprocess(data, channel):
    if channel in COLORS:
        return np.round(data * 255.0)
    return data

class PointCloud:
    """
    An array of points sampled on a surface. Each point may have zero or more
    channel attributes.

    :param coords: an [N x 3] array of point coordinates.
    :param channels: a dict mapping names to [N] arrays of channel values.
    """

    coords: np.ndarray
    channels: Dict[str, np.ndarray]

    def __init__(self, coords, channels):
        self.coords = coords
        self.channels = channels

    def __repr__(self) -> str:
        str = f'coords:{len(self.coords)} \t channels:{list(self.channels.keys())}'
        return str

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.coords,
            rgb=(
                np.stack([self.channels[x] for x in "RGB"], axis=1)
                if all(x in self.channels for x in "RGB")
                else None
            ),
        )

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return PointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"},
            )
        
    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with open(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx
            cur_dists = np.minimum(cur_dists, compute_dists(idx))
        return self.subsample(indices, **subsample_kwargs)

    def subsample(self, indices: np.ndarray, average_neighbors: bool = False) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
            )

        new_coords = self.coords[indices]
        neighbor_indices = PointCloud(coords=new_coords, channels={}).nearest_points(self.coords)

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(coords=new_coords, channels=new_channels)

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack([preprocess(self.channels[name], name) for name in channel_names], axis=-1)
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            dists = norms + np.sum(batch**2, axis=-1)[:, None] - 2 * (batch @ self.coords.T)
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0) for k, v in self.channels.items()
            },
        )



import io
import struct
from contextlib import contextmanager
from typing import BinaryIO, Iterator, Optional
import numpy as np

"""
adpoted from https://github.com/openai/point-e/blob/main/point_e/util/ply_util.py
"""
def write_ply(
    raw_f: BinaryIO,
    coords: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    faces: Optional[np.ndarray] = None,
):
    """
    Write a PLY file for a mesh or a point cloud.

    :param coords: an [N x 3] array of floating point coordinates.
    :param rgb: an [N x 3] array of vertex colors, in the range [0.0, 1.0].
    :param faces: an [N x 3] array of triangles encoded as integer indices.
    """
    with buffered_writer(raw_f) as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(bytes(f"element vertex {len(coords)}\n", "ascii"))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        if rgb is not None:
            f.write(b"property uchar red\n")
            f.write(b"property uchar green\n")
            f.write(b"property uchar blue\n")
        if faces is not None:
            f.write(bytes(f"element face {len(faces)}\n", "ascii"))
            f.write(b"property list uchar int vertex_index\n")
        f.write(b"end_header\n")

        if rgb is not None:
            rgb = (rgb * 255.499).round().astype(int)
            vertices = [
                (*coord, *rgb)
                for coord, rgb in zip(
                    coords.tolist(),
                    rgb.tolist(),
                )
            ]
            format = struct.Struct("<3f3B")
            for item in vertices:
                f.write(format.pack(*item))
        else:
            format = struct.Struct("<3f")
            for vertex in coords.tolist():
                f.write(format.pack(*vertex))

        if faces is not None:
            format = struct.Struct("<B3I")
            for tri in faces.tolist():
                f.write(format.pack(len(tri), *tri))


@contextmanager
def buffered_writer(raw_f: BinaryIO) -> Iterator[io.BufferedIOBase]:
    if isinstance(raw_f, io.BufferedIOBase):
        yield raw_f
    else:
        f = io.BufferedWriter(raw_f)
        yield f
        f.flush()