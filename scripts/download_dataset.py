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

    # Import kaggle here to avoid auth errors at startup
    try:
        import os

        # Map user's custom KAGGLE_API_TOKEN to standard KAGGLE_KEY if provided
        if os.getenv("KAGGLE_API_TOKEN") and not os.getenv("KAGGLE_KEY"):
            os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_API_TOKEN")

        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("❌ Kaggle API not installed. Run 'pip install kaggle'.")
        return

    api = KaggleApi()
    api.authenticate()

    print(f"Downloading {dataset_slug} to {target_dir}...")
    try:
        api.dataset_download_files(dataset_slug, path=str(target_dir), unzip=True)
        print("Download step finished.")
    except ValueError as e:
        if "does not match expected size" in str(e):
            print(f"Size mismatch error (Kaggle bug): {e}. Will attempt manual extraction.")
        else:
            print(f"❌ Error downloading dataset: {e}")
            raise
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        raise

    # Always manually extract any zip files that were left behind
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
