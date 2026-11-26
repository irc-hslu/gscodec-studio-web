import random
import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colormaps
from plyfile import PlyData


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_shape = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_shape, -1)
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_shape, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = []
        layers.append(
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width)
        )
        layers.append(torch.nn.ReLU(inplace=True))
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    # try:
    #     torch.use_deterministic_algorithms(True)
    # except Exception as e:
    #     print(f"Could not use deterministic algorithms: {e}")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def verify_random_seed(result_dir: str):
    """
    Verify random seed consistency by generating a checksum and save to file
    Should be called after set_random_seed()
    
    Args:
        result_dir: Directory to save the checksum file
    
    Returns:
        str: MD5 checksum of generated random numbers
    """
    import hashlib
    import os
    
    # Generate random numbers from different generators
    test_results = {
        'python_random': random.random(),
        'numpy_random': np.random.rand(),
        'torch_random': torch.rand(1).item(),
    }
    
    # Test CUDA random numbers if available
    if torch.cuda.is_available():
        test_results['torch_cuda_random'] = torch.cuda.FloatTensor(1).uniform_().item()
    
    # Generate checksum from results
    result_str = ""
    for key in sorted(test_results.keys()):
        result_str += f"{key}:{test_results[key]:.10f};"
    
    checksum = hashlib.md5(result_str.encode()).hexdigest()
    
    # Write checksum to file
    checksum_file = os.path.join(result_dir, 'seed_checksum.txt')
    with open(checksum_file, 'w') as f:
        f.write(checksum)
    
    print(f"Random seed checksum: {checksum}")
    print(f"Checksum saved to: {checksum_file}")
    print("Sample results:", test_results)
    
    return checksum


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor = None,
    near_plane: float = None,
    far_plane: float = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth, colormap="turbo")
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img

def save_ply(splats: torch.nn.ParameterDict, path: str):
    from plyfile import PlyData, PlyElement

    means = splats["means"].detach().cpu().numpy()
    normals = np.zeros_like(means)
    sh0 = splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    shN = splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = splats["opacities"].detach().unsqueeze(1).cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = splats["quats"].detach().cpu().numpy()

    def construct_list_of_attributes(splats):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']

        for i in range(splats["sh0"].shape[1]*splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(splats["shN"].shape[1]*splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
        return l

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(splats)]

    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = np.concatenate((means, normals, sh0, shN, opacities, scales, quats), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def load_ply(path: str) -> torch.nn.ParameterDict:
    # Read PLY file
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    
    # Get total number of vertices
    n_vertices = vertices.count

    # Extract basic attributes (positions)
    means = np.stack((vertices['x'], vertices['y'], vertices['z']), axis=1)
    
    # Calculate dimensions for sh0 and shN
    sh0_size = len([prop for prop in vertices.properties if prop.name.startswith('f_dc_')])
    shN_size = len([prop for prop in vertices.properties if prop.name.startswith('f_rest_')])
    
    # Extract sh0 data
    sh0_data = np.zeros((n_vertices, sh0_size))
    for i in range(sh0_size):
        sh0_data[:, i] = vertices[f'f_dc_{i}']
    
    # Extract shN data
    shN_data = np.zeros((n_vertices, shN_size))
    for i in range(shN_size):
        shN_data[:, i] = vertices[f'f_rest_{i}']
    
    # Extract opacity data
    opacities = vertices['opacity'].reshape(-1, 1)
    
    # Extract scales data
    scale_size = len([prop for prop in vertices.properties if prop.name.startswith('scale_')])
    scales = np.zeros((n_vertices, scale_size))
    for i in range(scale_size):
        scales[:, i] = vertices[f'scale_{i}']
    
    # Extract quaternion data
    quat_size = len([prop for prop in vertices.properties if prop.name.startswith('rot_')])
    quats = np.zeros((n_vertices, quat_size))
    for i in range(quat_size):
        quats[:, i] = vertices[f'rot_{i}']
    
    # Reshape sh0 and shN to [N, n_coeffs, 3]
    # PLY stores channel-major flattened coeffs; degree-0 assets have no f_rest_*.
    sh0_channels = 3
    if sh0_size == 0:
        sh0_data = np.zeros((n_vertices, 0, 3), dtype=np.float32)
    elif sh0_size % sh0_channels != 0:
        raise ValueError(f"Unexpected f_dc_* count {sh0_size} in {path}")
    else:
        sh0_coeffs = sh0_size // sh0_channels
        sh0_data = sh0_data.reshape(-1, sh0_channels, sh0_coeffs).transpose(0, 2, 1)

    shN_channels = 3
    if shN_size == 0:
        shN_data = np.zeros((n_vertices, 0, 3), dtype=np.float32)
    elif shN_size % shN_channels != 0:
        raise ValueError(f"Unexpected f_rest_* count {shN_size} in {path}")
    else:
        shN_coeffs = shN_size // shN_channels
        shN_data = shN_data.reshape(-1, shN_channels, shN_coeffs).transpose(0, 2, 1)
    
    # Convert to torch tensors and create ParameterDict
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(means.astype(np.float32))),
        "sh0": torch.nn.Parameter(torch.from_numpy(sh0_data.astype(np.float32))),
        "shN": torch.nn.Parameter(torch.from_numpy(shN_data.astype(np.float32))),
        "opacities": torch.nn.Parameter(torch.from_numpy(opacities.astype(np.float32)).squeeze(1)),
        "scales": torch.nn.Parameter(torch.from_numpy(scales.astype(np.float32))),
        "quats": torch.nn.Parameter(torch.from_numpy(quats.astype(np.float32)))
    })
    
    return splats


def concat_splat_dicts(splats_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """Concatenate splat dicts along the Gaussian axis, padding SH bands if needed."""
    if not splats_list:
        raise ValueError("concat_splat_dicts() requires at least one splat set")
    if len(splats_list) == 1:
        return splats_list[0]

    max_sh0 = max(int(s["sh0"].shape[1]) for s in splats_list)
    max_shN = max(int(s["shN"].shape[1]) for s in splats_list)
    padded = []
    for splats in splats_list:
        n = splats["means"].shape[0]
        device = splats["means"].device
        out = {k: v for k, v in splats.items()}
        cur0 = int(out["sh0"].shape[1])
        if cur0 < max_sh0:
            pad = torch.zeros(n, max_sh0 - cur0, 3, device=device, dtype=out["sh0"].dtype)
            out["sh0"] = torch.cat([out["sh0"], pad], dim=1)
        curN = int(out["shN"].shape[1])
        if curN < max_shN:
            pad = torch.zeros(n, max_shN - curN, 3, device=device, dtype=out["shN"].dtype)
            out["shN"] = torch.cat([out["shN"], pad], dim=1)
        padded.append(out)

    return {
        key: torch.cat([s[key] for s in padded], dim=0)
        for key in padded[0].keys()
    }


def trim_splat_padding(
    splat: Dict[str, Tensor],
    n_orig: Optional[int],
    pad_log_scale: float = -9.7,
) -> Dict[str, Tensor]:
    """Drop square-grid padding Gaussians and optionally trim to the original count."""
    scales = splat["scales"]
    keep = ~((scales < pad_log_scale).all(dim=-1))
    if not bool(keep.all()):
        splat = {key: value[keep] for key, value in splat.items()}
    if n_orig is not None and int(splat["means"].shape[0]) > n_orig:
        splat = {key: value[:n_orig] for key, value in splat.items()}
    return splat


def concat_splats(splats_list: List[torch.nn.ParameterDict]) -> torch.nn.ParameterDict:
    """Concatenate splat ParameterDicts along the Gaussian axis, padding SH bands if needed."""
    if not splats_list:
        raise ValueError("concat_splats() requires at least one splat set")
    if len(splats_list) == 1:
        return splats_list[0]

    max_sh0 = max(int(s["sh0"].shape[1]) for s in splats_list)
    max_shN = max(int(s["shN"].shape[1]) for s in splats_list)
    padded = [_pad_sh_bands(s, max_sh0, max_shN) for s in splats_list]
    return torch.nn.ParameterDict({
        key: torch.nn.Parameter(torch.cat([s[key].detach() for s in padded], dim=0))
        for key in padded[0].keys()
    })


def _pad_sh_bands(
    splats: torch.nn.ParameterDict, sh0_bands: int, shN_bands: int
) -> torch.nn.ParameterDict:
    n = splats["means"].shape[0]
    device = splats["means"].device
    out = torch.nn.ParameterDict({k: v for k, v in splats.items()})

    cur0 = int(out["sh0"].shape[1])
    if cur0 < sh0_bands:
        pad = torch.zeros(n, sh0_bands - cur0, 3, device=device, dtype=out["sh0"].dtype)
        out["sh0"] = torch.nn.Parameter(torch.cat([out["sh0"].detach(), pad], dim=1))

    curN = int(out["shN"].shape[1])
    if curN < shN_bands:
        pad = torch.zeros(n, shN_bands - curN, 3, device=device, dtype=out["shN"].dtype)
        out["shN"] = torch.nn.Parameter(torch.cat([out["shN"].detach(), pad], dim=1))

    return out


def find_3dgs_iteration_dir(ply_dir: str, iteration: Optional[int] = None) -> Optional[str]:
    """Return a 3DGS `point_cloud/iteration_*` directory if ply_dir is a training export."""
    if not ply_dir or not os.path.isdir(ply_dir):
        return None

    def _has_scene_ply(directory: str) -> bool:
        if os.path.isfile(os.path.join(directory, "point_cloud.ply")):
            return True
        return bool(glob.glob(os.path.join(directory, "point_cloud_*.ply")))

    if _has_scene_ply(ply_dir):
        return ply_dir

    pattern = os.path.join(ply_dir, "point_cloud", "iteration_*")
    iter_dirs = [p for p in glob.glob(pattern) if os.path.isdir(p)]
    if not iter_dirs:
        nested = os.path.join(ply_dir, "point_cloud")
        if os.path.isdir(nested):
            iter_dirs = [
                p for p in glob.glob(os.path.join(nested, "iteration_*")) if os.path.isdir(p)
            ]
    if not iter_dirs:
        return None

    def _iter_index(path: str) -> int:
        return int(os.path.basename(path).rsplit("_", 1)[-1])

    if iteration is not None:
        target = os.path.join(os.path.dirname(iter_dirs[0]), f"iteration_{iteration}")
        if not os.path.isdir(target):
            raise FileNotFoundError(f"No point_cloud/iteration_{iteration} under {ply_dir}")
        return target

    return max(iter_dirs, key=_iter_index)


def list_scene_lod_levels(iteration_dir: str) -> List[int]:
    """Return sorted LOD indices found as ``point_cloud_<n>.ply`` (e.g. 1, 2, 3)."""
    levels = []
    for path in glob.glob(os.path.join(iteration_dir, "point_cloud_*.ply")):
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = stem[len("point_cloud_") :]
        if suffix.isdigit():
            levels.append(int(suffix))
    return sorted(levels)


def resolve_scene_ply_path(
    iteration_dir: str, ply_lod: Optional[int] = None
) -> str:
    """Pick the scene PLY: ``point_cloud.ply`` or ``point_cloud_<lod>.ply``.

    - ``ply_lod=None``: prefer ``point_cloud.ply``; if missing, use the densest LOD
      (lowest index among ``point_cloud_*.ply``) and print a notice.
    - ``ply_lod=N``: require ``point_cloud_N.ply``.
    """
    full_path = os.path.join(iteration_dir, "point_cloud.ply")
    lod_levels = list_scene_lod_levels(iteration_dir)

    if ply_lod is not None:
        lod_path = os.path.join(iteration_dir, f"point_cloud_{ply_lod}.ply")
        if not os.path.isfile(lod_path):
            available = ", ".join(str(x) for x in lod_levels) or "none"
            raise FileNotFoundError(
                f"Missing {lod_path}. Available LOD files: {available}. "
                f"Pass --ply-lod with one of those indices."
            )
        return lod_path

    if os.path.isfile(full_path):
        return full_path

    if lod_levels:
        chosen = lod_levels[0]
        lod_path = os.path.join(iteration_dir, f"point_cloud_{chosen}.ply")
        print(
            f"No point_cloud.ply; using LOD point_cloud_{chosen}.ply "
            f"(available LODs: {lod_levels}). Pass --ply-lod N to pick another."
        )
        return lod_path

    raise FileNotFoundError(
        f"No point_cloud.ply or point_cloud_*.ply under {iteration_dir}"
    )


def load_3dgs_iteration(
    iteration_dir: str,
    merge_environment: bool = False,
    ply_lod: Optional[int] = None,
) -> Tuple[torch.nn.ParameterDict, Optional[torch.nn.ParameterDict]]:
    """Load scene ``point_cloud[.N].ply`` and optionally ``environment.ply`` as layers."""
    main_path = resolve_scene_ply_path(iteration_dir, ply_lod)
    env_path = os.path.join(iteration_dir, "environment.ply")

    env = None
    if merge_environment and os.path.isfile(env_path):
        print(f"Loading environment GS: {env_path}")
        env = load_ply(env_path)
    elif os.path.isfile(env_path) and not merge_environment:
        print(f"Skipping environment GS ({env_path}); pass --merge-environment to include it.")

    print(f"Loading main GS: {main_path}")
    scene = load_ply(main_path)
    n_scene = int(scene["means"].shape[0])
    n_env = int(env["means"].shape[0]) if env is not None else 0
    print(
        f"Loaded scene={n_scene} env={n_env} Gaussians "
        f"(sh0={tuple(scene['sh0'].shape[1:])}, shN={tuple(scene['shN'].shape[1:])})"
    )
    return scene, env