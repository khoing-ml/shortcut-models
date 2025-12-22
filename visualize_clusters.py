#!/usr/bin/env python3
"""
Visualize clustering results by showing sample images from each cluster.

Reads the output from encode_and_cluster.py and creates visualizations:
- Grid images showing samples from each cluster
- Optional HTML page with all clusters
- Cluster statistics

Example:
  python visualize_clusters.py --cluster-dir encode_cluster_out --max-per-cluster 16 --output-dir cluster_viz
  
  # For dataset iterator mode (with latent decoding):
  python visualize_clusters.py --cluster-dir encode_cluster_out --output-dir cluster_viz \
    --use-latents --dataset-name celebahq256
"""
import argparse
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("Warning: matplotlib not found. Install with: pip install matplotlib")
    plt = None

try:
    from PIL import Image
except ImportError:
    print("Warning: PIL not found. Install with: pip install Pillow")
    Image = None


def load_vae_and_dataset(dataset_name, batch_size=64):
    """Load VAE decoder and dataset iterator for latent decoding."""
    try:
        import jax
        import jax.numpy as jnp
        from utils.stable_vae import StableVAE
        from utils.datasets import get_dataset
    except ImportError as e:
        raise ImportError("JAX and project utils are required for latent decoding") from e
    
    vae = StableVAE.create()
    # Use the decode method directly - it's already jitted with static_argnames in the class
    vae_decode = vae.decode
    
    # Load dataset iterator
    dataset_iter = get_dataset(dataset_name, batch_size, is_train=True, debug_overfit=False)
    
    return vae, vae_decode, dataset_iter


def decode_latent_to_image(vae_decode, latent, image_size=(256, 256)):
    """Decode a single latent to an image array."""
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX is required for latent decoding")
    
    # latent shape should be (lh, lw, 4), need to add batch dimension
    if latent.ndim == 1:
        # Assume 64x64x4 latent space for 512x512 images, or 32x32x4 for 256x256
        if image_size[0] == 512:
            latent = latent.reshape(64, 64, 4)
        else:  # 256x256
            latent = latent.reshape(32, 32, 4)
    
    latent_batch = jnp.array(latent[None, ...])  # Add batch dimension
    decoded = vae_decode(latent_batch, scale=True)  # Output shape: (1, h, w, 3)
    decoded_np = np.array(decoded[0])  # Remove batch dimension
    
    # Convert from [-1, 1] to [0, 255]
    image = ((decoded_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
    
    # Resize if needed
    if image.shape[:2] != image_size:
        if Image is not None:
            image_pil = Image.fromarray(image)
            image_pil = image_pil.resize(image_size, Image.Resampling.LANCZOS)
            image = np.array(image_pil)
    
    return image


def load_assignments(csv_path):
    """Load cluster assignments from CSV file."""
    assignments = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row['path']
            cluster = int(row['cluster'])
            assignments[path] = cluster
    return assignments


def group_by_cluster(assignments):
    """Group image paths by cluster ID."""
    clusters = defaultdict(list)
    for path, cluster_id in assignments.items():
        clusters[cluster_id].append(path)
    return clusters


def load_image(image_path, size=(128, 128), latents=None, vae_decode=None, path_to_index=None):
    """Load and resize an image.
    
    Args:
        image_path: Path to image file or synthetic path like "sample_0"
        size: Target size for the image
        latents: Optional latents array for decoding
        vae_decode: Optional VAE decoder function
        path_to_index: Optional mapping from path to index in latents array
    """
    if Image is None:
        raise ImportError("PIL is required for image loading")
    
    # Convert Path object to string for lookup
    path_str = str(image_path)
    
    # Check if this is a synthetic path (e.g., "sample_0") and we have latents
    if latents is not None and vae_decode is not None and path_to_index is not None:
        if path_str in path_to_index:
            idx = path_to_index[path_str]
            latent = latents[idx]
            try:
                return decode_latent_to_image(vae_decode, latent, image_size=size)
            except Exception as e:
                print(f"Warning: Could not decode latent for {path_str}: {e}")
                return np.zeros((*size, 3), dtype=np.uint8)
    
    # Otherwise, try to load as a regular image file
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(size, Image.Resampling.LANCZOS)
        return np.array(img)
    except Exception as e:
        print(f"Warning: Could not load {image_path}: {e}")
        # Return a placeholder image
        return np.zeros((*size, 3), dtype=np.uint8)


def visualize_cluster_grid(cluster_id, image_paths, max_images=16, image_size=(128, 128), 
                          data_dir=None, output_path=None, latents=None, vae_decode=None, path_to_index=None):
    """Create a grid visualization for a single cluster."""
    if plt is None:
        raise ImportError("matplotlib is required for visualization")
    
    n_images = min(len(image_paths), max_images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    fig.suptitle(f'Cluster {cluster_id} ({len(image_paths)} images)', fontsize=14, fontweight='bold')
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        if i < n_images:
            img_path = image_paths[i]
            # Pass the original string path for latent lookup
            img = load_image(img_path, size=image_size, latents=latents, 
                           vae_decode=vae_decode, path_to_index=path_to_index)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_all_clusters_overview(clusters, data_dir=None, output_path=None, 
                                     samples_per_cluster=4, image_size=(64, 64),
                                     latents=None, vae_decode=None, path_to_index=None):
    """Create an overview visualization showing samples from all clusters."""
    if plt is None:
        raise ImportError("matplotlib is required for visualization")
    
    n_clusters = len(clusters)
    cluster_ids = sorted(clusters.keys())
    
    n_cols = samples_per_cluster
    n_rows = n_clusters
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    fig.suptitle(f'Cluster Overview ({n_clusters} clusters)', fontsize=16, fontweight='bold')
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row, cluster_id in enumerate(cluster_ids):
        image_paths = clusters[cluster_id][:samples_per_cluster]
        
        for col in range(n_cols):
            ax = axes[row, col]
            
            if col < len(image_paths):
                img_path = image_paths[col]
                # Pass the original string path for latent lookup
                img = load_image(img_path, size=image_size, latents=latents,
                               vae_decode=vae_decode, path_to_index=path_to_index)
                ax.imshow(img)
            
            ax.axis('off')
            
            # Add cluster ID label on the first image of each row
            if col == 0:
                ax.text(-0.1, 0.5, f'C{cluster_id}\n({len(clusters[cluster_id])})', 
                       transform=ax.transAxes, fontsize=8, va='center', ha='right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_cluster_statistics(clusters, output_path=None):
    """Create and save cluster statistics."""
    cluster_ids = sorted(clusters.keys())
    sizes = [len(clusters[cid]) for cid in cluster_ids]
    
    stats = {
        'n_clusters': len(clusters),
        'total_images': sum(sizes),
        'mean_size': np.mean(sizes),
        'median_size': np.median(sizes),
        'min_size': np.min(sizes),
        'max_size': np.max(sizes),
        'std_size': np.std(sizes)
    }
    
    print("\n" + "="*50)
    print("CLUSTER STATISTICS")
    print("="*50)
    print(f"Number of clusters: {stats['n_clusters']}")
    print(f"Total images: {stats['total_images']}")
    print(f"Average cluster size: {stats['mean_size']:.1f}")
    print(f"Median cluster size: {stats['median_size']:.1f}")
    print(f"Min cluster size: {stats['min_size']}")
    print(f"Max cluster size: {stats['max_size']}")
    print(f"Std dev of cluster size: {stats['std_size']:.1f}")
    print("="*50 + "\n")
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write("CLUSTER STATISTICS\n")
            f.write("="*50 + "\n")
            for key, value in stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n\nCluster sizes:\n")
            for cid in cluster_ids:
                f.write(f"Cluster {cid}: {len(clusters[cid])} images\n")
    
    # Create histogram of cluster sizes
    if plt is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(sizes, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Cluster Size', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Distribution of Cluster Sizes', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        
        if output_path:
            hist_path = Path(output_path).parent / "cluster_size_histogram.png"
            plt.savefig(hist_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"Saved histogram to {hist_path}")
        else:
            plt.show()
    
    return stats


def generate_html_report(clusters, output_dir, data_dir=None, max_per_cluster=16):
    """Generate an HTML report with all clusters."""
    html_path = Path(output_dir) / "cluster_report.html"
    cluster_ids = sorted(clusters.keys())
    
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Cluster Visualization Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .cluster {
            background-color: white;
            margin: 20px 0;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .cluster-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #2c3e50;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(128px, 1fr));
            gap: 10px;
        }
        .image-grid img {
            width: 100%;
            height: 128px;
            object-fit: cover;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .stats {
            background-color: #e8f4f8;
            padding: 10px;
            margin: 20px 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
    </style>
</head>
<body>
    <h1>Cluster Visualization Report</h1>
    <div class="stats">
        <strong>Total Clusters:</strong> {n_clusters}<br>
        <strong>Total Images:</strong> {total_images}
    </div>
""".format(n_clusters=len(clusters), total_images=sum(len(clusters[c]) for c in cluster_ids))
    
    for cluster_id in cluster_ids:
        image_paths = clusters[cluster_id][:max_per_cluster]
        html_content += f"""
    <div class="cluster">
        <div class="cluster-header">Cluster {cluster_id} ({len(clusters[cluster_id])} images)</div>
        <div class="image-grid">
"""
        for img_path in image_paths:
            # For HTML, we'll need relative paths
            if data_dir:
                try:
                    rel_path = Path(img_path).relative_to(Path(data_dir).parent)
                except ValueError:
                    rel_path = img_path
            else:
                rel_path = img_path
            html_content += f'            <img src="{rel_path}" alt="Image from cluster {cluster_id}">\n'
        
        html_content += """        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated HTML report at {html_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize clustering results")
    p.add_argument("--cluster-dir", required=True, help="Directory containing assignments.csv and clustering results")
    p.add_argument("--data-dir", default=None, help="Root directory containing the images (if paths in CSV are relative)")
    p.add_argument("--output-dir", default="cluster_viz", help="Output directory for visualizations")
    p.add_argument("--max-per-cluster", type=int, default=16, help="Maximum number of images to show per cluster")
    p.add_argument("--image-size", type=int, default=256, help="Size to display images (width and height)")
    p.add_argument("--overview-samples", type=int, default=4, help="Number of samples per cluster in overview")
    p.add_argument("--generate-html", action="store_true", help="Generate an HTML report")
    p.add_argument("--specific-clusters", type=int, nargs='+', help="Visualize only specific cluster IDs")   
    p.add_argument("--use-latents", action="store_true", help="Decode images from latents (for dataset iterator mode)")
    p.add_argument("--dataset-name", default=None, help="Dataset name (required if --use-latents is set)")    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load assignments
    csv_path = Path(args.cluster_dir) / "assignments.csv"
    if not csv_path.exists():
        print(f"Error: assignments.csv not found in {args.cluster_dir}")
        return
    
    print(f"Loading assignments from {csv_path}...")
    assignments = load_assignments(csv_path)
    clusters = group_by_cluster(assignments)
    
    print(f"Found {len(clusters)} clusters with {len(assignments)} total images")
    
    # Load latents and VAE if needed
    latents = None
    vae_decode = None
    path_to_index = None
    
    if args.use_latents:
        if args.dataset_name is None:
            print("Error: --dataset-name is required when using --use-latents")
            return
        
        # Load latents
        latents_path = Path(args.cluster_dir) / "latents.npy"
        if not latents_path.exists():
            print(f"Error: latents.npy not found in {args.cluster_dir}")
            return
        
        print(f"Loading latents from {latents_path}...")
        latents = np.load(latents_path)
        print(f"Loaded latents with shape {latents.shape}")
        
        # Create path to index mapping
        path_to_index = {path: idx for idx, path in enumerate(assignments.keys())}
        
        # Load VAE decoder
        print(f"Loading VAE decoder for dataset {args.dataset_name}...")
        vae, vae_decode, _ = load_vae_and_dataset(args.dataset_name)
        print("VAE loaded successfully")
    
    # Generate statistics
    stats_path = output_dir / "cluster_statistics.txt"
    create_cluster_statistics(clusters, output_path=stats_path)
    
    # Determine which clusters to visualize
    if args.specific_clusters:
        cluster_ids = [cid for cid in args.specific_clusters if cid in clusters]
        if not cluster_ids:
            print("Error: None of the specified cluster IDs exist")
            return
        print(f"Visualizing specific clusters: {cluster_ids}")
    else:
        cluster_ids = sorted(clusters.keys())
        print(f"Visualizing all {len(cluster_ids)} clusters")
    
    # Create individual cluster visualizations
    print("\nGenerating individual cluster visualizations...")
    cluster_output_dir = output_dir / "individual_clusters"
    cluster_output_dir.mkdir(parents=True, exist_ok=True)
    
    for cluster_id in cluster_ids:
        output_path = cluster_output_dir / f"cluster_{cluster_id:04d}.png"
        print(f"  Cluster {cluster_id}...", end=" ")
        visualize_cluster_grid(
            cluster_id, 
            clusters[cluster_id], 
            max_images=args.max_per_cluster,
            image_size=(args.image_size, args.image_size),
            data_dir=args.data_dir,
            output_path=output_path,
            latents=latents,
            vae_decode=vae_decode,
            path_to_index=path_to_index
        )
        print(f"saved to {output_path}")
    
    # Create overview visualization (only if not too many clusters)
    if not args.specific_clusters and len(clusters) <= 50:
        print("\nGenerating overview visualization...")
        overview_path = output_dir / "clusters_overview.png"
        visualize_all_clusters_overview(
            clusters,
            data_dir=args.data_dir,
            output_path=overview_path,
            samples_per_cluster=args.overview_samples,
            image_size=(args.image_size // 2, args.image_size // 2),
            latents=latents,
            vae_decode=vae_decode,
            path_to_index=path_to_index
        )
        print(f"Saved overview to {overview_path}")
    elif len(clusters) > 50:
        print("\nSkipping overview visualization (too many clusters). Use --specific-clusters to visualize a subset.")
    
    # Generate HTML report if requested
    if args.generate_html:
        print("\nGenerating HTML report...")
        generate_html_report(clusters, output_dir, data_dir=args.data_dir, max_per_cluster=args.max_per_cluster)
    
    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
