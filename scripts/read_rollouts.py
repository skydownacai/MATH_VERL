import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rollout_path", type=str, default="./outputs/rollout_data", help="Path to the rollout data directory.")
parser.add_argument("--project_name", type=str, help="Name of the project to read rollouts from.", default="DeepScaleR-grpo")
parser.add_argument("--run_name", type=str, help="Name of the run to read rollouts from.", default="DeepScaleR-grpo-20260305-082905")
parser.add_argument("--json_path", type=str, help="Path to the JSON file containing the rollout data.", default=None)
parser.add_argument("--step", type=int, help="Step number of the rollout to read.", default=1)
parser.add_argument("--index", type=int, help="Index of the trajectory to read.", default=0)
args = parser.parse_args()

if not args.json_path:
    json_path = os.path.join(args.rollout_path, args.project_name, args.run_name, f"{args.step}.jsonl")
else:
    json_path = args.json_path

# Read the jsonl file. 
with open(json_path, 'r', encoding='utf-8') as f:
    rollout_data = [json.loads(line) for line in f if line.strip()]

# Print the trajectory at the specified index.
if args.index < len(rollout_data):
    trajectory = rollout_data[args.index]
    print(json.dumps(trajectory, indent=2))
else:    print(f"Index {args.index} is out of range. Total trajectories: {len(rollout_data)}")