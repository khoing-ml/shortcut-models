#!/usr/bin/env python3
"""
Encode images into a latent space and cluster them with k-means.

Supports two encoders:
- `stablevae`: uses the repository's `utils.stable_vae.StableVAE` (Flax/JAX)
- `resnet`: uses torchvision ResNet50 (PyTorch)

Example:
  python encode_and_cluster.py --data-dir /path/to/images --out-dir results --n-clusters 50 --encoder stablevae

Outputs:
- `latents.npy` (N x D)
- `cluster_centers.npy`
- `assignments.csv`
- `examples/cluster_XXXX/` sample images per cluster
"""
from pathlib import Path
import argparse
import csv
import shutil
from collections import defaultdict

import numpy as np

from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def build_imagefolder_dataset(data_dir, image_size=224, batch_size=64, num_workers=4):
    try:
        import torch
        import torchvision.transforms as T
        import torchvision.datasets as datasets
    except Exception as e:
        raise ImportError("torch and torchvision are required for data loading. Install them with pip or conda.") from e

    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = __import__("torch").utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=num_workers, pin_memory=True)
    return dataset, loader


def encode_with_stablevae(loader, dataset, image_size=512, batch_size=8, seed=0):
    # Uses repo's Flax StableVAE
    import jax
    import jax.numpy as jnp
    from utils.stable_vae import StableVAE

    vae = StableVAE.create()
    latents_list = []
    for images, _ in tqdm(loader, desc="Encoding (stablevae)"):
        # images is a torch tensor in [0,1] shape (B,C,H,W)
        imgs_t = images.cpu().numpy()
        # convert to channels-last and to float32
        imgs_cl = np.transpose(imgs_t, (0, 2, 3, 1)).astype(np.float32)
        key = jax.random.PRNGKey(seed)
        seed += 1
        imgs_jax = jnp.array(imgs_cl)
        lat = vae.encode(key, imgs_jax, scale=True)  # (B, lh, lw, 4)
        lat_np = np.array(jax.device_get(lat))
        B = lat_np.shape[0]
        lat_np = lat_np.reshape(B, -1)
        latents_list.append(lat_np)
    latents = np.concatenate(latents_list, axis=0)
    paths = [p for (p, _) in dataset.samples]
    return latents, paths


def cluster_latents(latents, n_clusters=100, batch_size=1024, random_state=42):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
    assignments = kmeans.fit_predict(latents)
    return kmeans, assignments


def save_results(out_dir, latents, paths, assignments, kmeans):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "latents.npy", latents)
    np.save(out / "cluster_centers.npy", kmeans.cluster_centers_)
    csv_path = out / "assignments.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "cluster"])
        for p, c in zip(paths, assignments):
            w.writerow([p, int(c)])
    return csv_path


def dump_cluster_examples(out_dir, paths, assignments, max_per_cluster=10):
    out = Path(out_dir)
    examples = out / "examples"
    examples.mkdir(exist_ok=True)
    clusters = defaultdict(list)
    for p, c in zip(paths, assignments):
        if len(clusters[c]) < max_per_cluster:
            clusters[c].append(p)
    for c, plist in clusters.items():
        cdir = examples / f"cluster_{c:04d}"
        cdir.mkdir(exist_ok=True)
        for src in plist:
            try:
                shutil.copy(src, cdir)
            except Exception:
                pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="ImageFolder formatted directory")
    p.add_argument("--out-dir", default="encode_cluster_out")
    p.add_argument("--n-clusters", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--image-size", type=int, default=512, help="Image size for encoder; stablevae prefers 512")
    p.add_argument("--encoder", choices=["stablevae", "resnet"], default="stablevae")
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-examples", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    dataset, loader = build_imagefolder_dataset(args.data_dir, image_size=args.image_size,
                                               batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"Found {len(dataset)} images")

    if args.encoder == "resnet":
        latents, paths = encode_with_resnet(loader, dataset, device=args.device)
    else:
        # stablevae works better with smaller batch sizes; override if user provided larger
        safe_batch = max(1, min(args.batch_size, 8))
        # rebuild loader with safe batch size
        dataset, loader = build_imagefolder_dataset(args.data_dir, image_size=args.image_size,
                                                   batch_size=safe_batch, num_workers=args.num_workers)
        latents, paths = encode_with_stablevae(loader, dataset, image_size=args.image_size, batch_size=safe_batch)

    print("Clustering...")
    kmeans, assignments = cluster_latents(latents, n_clusters=args.n_clusters)
    csv_path = save_results(args.out_dir, latents, paths, assignments, kmeans)
    dump_cluster_examples(args.out_dir, paths, assignments, max_per_cluster=args.max_examples)
    print(f"Saved assignments to {csv_path}")


if __name__ == '__main__':
    main()
