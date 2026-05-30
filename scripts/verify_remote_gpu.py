import argparse
import sys
import os

from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.utils.remote_gpu import GPUManager
except ImportError:
    print(
        "Error: Could not import GPUManager. Ensure you are running from the project root."
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify remote Kaggle GPU connection")
    parser.add_argument(
        "--json", default="gpu_connection.json", help="Path to gpu_connection.json"
    )
    parser.add_argument(
        "--key", default="~/.ssh/id_ed25519", help="Path to your SSH private key"
    )
    args = parser.parse_args()

    if not os.path.exists(os.path.expanduser(args.json)):
        print(f"Error: {args.json} not found. Did you download it from Kaggle?")
        sys.exit(1)

    mgr = GPUManager()

    try:
        print(f"Loading connection info from {args.json}...")
        mgr.add_backend_from_json("kaggle", args.json)

        # Override key path if specified
        mgr._backends["kaggle"].ssh_key = args.key

        print("Attempting to connect to Kaggle GPU...")
        with mgr.use("kaggle") as gpu:
            print("Connection established!")

            print("\nRemote GPU Information:")
            print(gpu.gpu_info())

            # Determine remote home directory
            home_res = gpu.run("echo $HOME", stream=False)
            remote_home = home_res.stdout.strip() if home_res.ok() else "/home/zeus"
            if not remote_home:
                remote_home = "/home/zeus"

            print("\nTesting file sync (creating remote directory)...")
            gpu.run(f"mkdir -p {remote_home}/test_sync")

            print("\nTesting simple write...")
            gpu.write_file(
                f"{remote_home}/test_sync/hello.txt", "Hello from local machine!"
            )

            print("\nVerification complete. You are ready to train!")

    except Exception as e:
        print(f"\nConnection failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure the Kaggle notebook is currently RUNNING.")
        print("2. Ensure you have installed 'cloudflared' and it is in your PATH.")
        print("3. Check that your SSH private key path is correct.")
        print("4. Verify that you added the CORRECT public key to Kaggle Secrets.")
        sys.exit(1)


if __name__ == "__main__":
    main()
