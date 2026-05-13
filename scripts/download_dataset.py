import argparse
from pathlib import Path
import sys

# Add project root to sys.path to allow importing src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config


def download_dataset(dry_run=False):
    """
    Download and extract the VIP Cup 2021 dataset from Kaggle.
    """
    config = load_config()
    dataset_slug = config.get("dataset", {}).get(
        "kaggle_slug", "vsharma1/ieee-vip-cup-2021"
    )
    target_dir = Path(config.get("dataset", {}).get("root", "data/raw"))

    print(f"Dataset: {dataset_slug}")
    print(f"Target:  {target_dir}")

    if dry_run:
        print("Dry run complete. No files downloaded.")
        return

    # Ensure target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if data is already present to skip download
    # We look for actual ground truth files which are required
    existing_samples = list(target_dir.glob("**/joints_gt_*.mat"))
    if existing_samples:
        print(f"✅ Dataset already appears to be present in {target_dir} ({len(existing_samples)} joint files found). Skipping download.")
        return

    import subprocess
    import os

    # Map user's custom KAGGLE_API_TOKEN to standard KAGGLE_KEY if provided
    if os.getenv("KAGGLE_API_TOKEN") and not os.getenv("KAGGLE_KEY"):
        os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN")

    print(f"Downloading {dataset_slug} to {target_dir} using Kaggle CLI...")
    try:
        # We use the CLI directly as the Python API sometimes fails silently or downloads corrupted zips
        cmd = ["kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(target_dir), "--unzip", "--force"]
        # Use subprocess.run with default stdout/stderr to stream output to parent process logs
        subprocess.run(cmd, check=True)
        print("Download step finished.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error downloading dataset via CLI (exit code {e.returncode})")
        raise
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

    # Always manually extract any zip files that were left behind if --unzip failed
    import zipfile
    for zip_path in target_dir.glob("*.zip"):
        print(f"Found {zip_path.name}, extracting...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            zip_path.unlink()
            print(f"Extracted and removed {zip_path.name}")
        except Exception as e:
            print(f"❌ Failed to extract {zip_path.name}: {e}")
            print(f"File size: {zip_path.stat().st_size} bytes")
            # If it's corrupted, we should probably delete it so the next run tries again
            zip_path.unlink(missing_ok=True)
            raise

    # Verify
    files = list(target_dir.glob("*"))
    print(f"Files in target directory: {[f.name for f in files]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download VIP Cup 2021 dataset from Kaggle."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print info without downloading."
    )
    args = parser.parse_args()

    download_dataset(dry_run=args.dry_run)
