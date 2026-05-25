import os
import urllib.request


def download_file(url, destination):
    print(f"Downloading {url} to {destination}...")
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    # Progress callback
    def report(block_num, block_size, total_size):
        read_so_far = block_num * block_size
        if total_size > 0:
            percent = read_so_far * 100 / total_size
            print(
                f"\rProgress: {percent:.2f}% ({read_so_far / (1024**2):.2f} MB / {total_size / (1024**2):.2f} MB)",
                end="",
            )
        else:
            print(f"\rDownloaded {read_so_far / (1024**2):.2f} MB", end="")

    try:
        urllib.request.urlretrieve(url, destination, reporthook=report)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nFailed to download: {e}")
        return False


if __name__ == "__main__":
    url = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-b-multi-coco.pth"
    dest = "pretrained_models/vitpose-b-multi-coco.pth"
    success = download_file(url, dest)
    if not success:
        # Fallback to single coco model
        url_single = "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/coco/vitpose-b-coco.pth"
        print(f"Trying fallback: {url_single}")
        download_file(url_single, dest)
