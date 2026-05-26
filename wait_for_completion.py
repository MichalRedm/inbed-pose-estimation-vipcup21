import subprocess
import time
import sys
import json

def get_status():
    try:
        result = subprocess.run(['python', 'scripts/api_cli.py', 'status'], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

def main():
    print("Starting monitoring...")
    while True:
        status_output = get_status()
        print(f"Current Status:\n{status_output}")
        
        if "Status: Stopped" in status_output:
            print("Training stopped.")
            break
        
        time.sleep(30) # Poll every 30 seconds

    # Final checks
    import os
    eval_file = 'results/runs/loop45_ghost_vitpose/evaluation_val.json'
    history_file = 'results/runs/loop45_ghost_vitpose/history.json'
    
    if os.path.exists(eval_file):
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
            print(f"FINAL_EVAL_DATA: {json.dumps(eval_data)}")
    else:
        print("Evaluation file not found.")

    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history_data = json.load(f)
            best_pck = max([h.get('val_pck', 0) for h in history_data])
            print(f"BEST_VAL_PCK: {best_pck}")
    else:
        print("History file not found.")

if __name__ == "__main__":
    main()
