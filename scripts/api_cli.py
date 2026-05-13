import argparse
import json
import sys
import requests

API_URL = "http://localhost:8000"


def start_training(run_id, config_path, remote, resume, auto_eval):
    payload = {
        "run_id": run_id,
        "config_path": config_path,
        "remote": remote,
        "auto_eval": auto_eval,
        "training": {"resume": resume} if resume else {},
    }
    try:
        response = requests.post(f"{API_URL}/training/start", json=payload)
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error starting training: {e}")
        sys.exit(1)


def stop_training():
    try:
        response = requests.post(f"{API_URL}/training/stop")
        response.raise_for_status()
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error stopping training: {e}")
        sys.exit(1)


def get_status():
    try:
        response = requests.get(f"{API_URL}/training/status")
        response.raise_for_status()
        data = response.json()
        # Print a nice summary instead of raw JSON
        status = "Running" if data.get("is_running") else "Stopped"
        print(f"Status: {status}")
        print(f"Run ID: {data.get('run_id', 'N/A')}")
        print(f"Epoch: {data.get('current_epoch', 0)}/{data.get('total_epochs', 0)}")
        print(f"Progress: {data.get('progress', 0.0) * 100:.1f}%")
        print(f"Message: {data.get('status_message', 'N/A')}")

        if data.get("loss_history"):
            print(f"Latest Loss: {data['loss_history'][-1]:.4f}")
    except Exception as e:
        print(f"Error getting status: {e}")
        sys.exit(1)


def get_logs(last_n=20):
    try:
        response = requests.get(f"{API_URL}/training/status")
        response.raise_for_status()
        logs = response.json().get("log_history", [])
        for line in logs[-last_n:]:
            print(line)
    except Exception as e:
        print(f"Error getting logs: {e}")
        sys.exit(1)


def monitor(interval=2):
    import time

    try:
        print(f"Monitoring training (Ctrl+C to stop)...")
        while True:
            response = requests.get(f"{API_URL}/training/status")
            response.raise_for_status()
            data = response.json()

            status = "Running" if data.get("is_running") else "Stopped"
            msg = data.get("status_message", "N/A")
            epoch = f"{data.get('current_epoch', 0)}/{data.get('total_epochs', 0)}"
            prog = data.get("progress", 0.0) * 100

            # Single line status update
            sys.stdout.write(
                f"\r[{status}] Epoch: {epoch} | Progress: {prog:5.1f}% | Msg: {msg[:50]:<50}"
            )
            sys.stdout.flush()

            if not data.get("is_running") and status == "Stopped":
                print("\nTraining stopped.")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\nError monitoring: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="In-Bed Pose API CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start
    start_parser = subparsers.add_parser("start", help="Start training")
    start_parser.add_argument("--run_id", type=str, required=True, help="Run ID")
    start_parser.add_argument("--config", type=str, help="Path to config YAML")
    start_parser.add_argument("--remote", action="store_true", help="Run on remote GPU")
    start_parser.add_argument(
        "--resume", action="store_true", help="Resume from checkpoint"
    )
    start_parser.add_argument(
        "--eval", action="store_true", help="Run evaluation after training"
    )

    # Stop
    subparsers.add_parser("stop", help="Stop training")

    # Status
    subparsers.add_parser("status", help="Get training status")

    # Logs
    logs_parser = subparsers.add_parser("logs", help="Get training logs")
    logs_parser.add_argument("-n", type=int, default=20, help="Number of lines to show")

    # Monitor
    monitor_parser = subparsers.add_parser("monitor", help="Monitor training progress")
    monitor_parser.add_argument(
        "-i", "--interval", type=int, default=2, help="Refresh interval in seconds"
    )

    args = parser.parse_args()

    if args.command == "start":
        start_training(args.run_id, args.config, args.remote, args.resume, args.eval)
    elif args.command == "stop":
        stop_training()
    elif args.command == "status":
        get_status()
    elif args.command == "logs":
        get_logs(args.n)
    elif args.command == "monitor":
        monitor(args.interval)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
