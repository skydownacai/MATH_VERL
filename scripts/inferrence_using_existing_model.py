import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./checkpoints/DeepScaleR-grpo/DeepScaleR-grpo-20260305-082905",\
    help="Path to the model checkpoint directory.")
parser.add_argument("--prompt", type=str, default="What is the derivative of sin(x)?", help="The prompt to feed into the model for inference.")
parser.add_argument("--global_step", type=int, default=-1, help="The global step number of the model checkpoint to load. If -1, load the latest checkpoint.")
parser.add_argument("--max_length", type=int, default=8096, help="The maximum length of the generated output.")
parser.add_argument("--device", type=str, default="cpu", help="The device to run the inference on. Use 'cuda' for GPU and 'cpu' for CPU.")
args = parser.parse_args()

# Device.
device = args.device
if device == "cuda":
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the huggingface checkpoints.
from transformers import AutoTokenizer, AutoModelForCausalLM
model_path = args.model_path
if args.global_step == -1:
    # read the latest_checkpointed_iteration.txt to get the latest step.
    with open(os.path.join(model_path, "latest_checkpointed_iteration.txt"), 'r') as f:
        latest_step = int(f.read().strip())
    step = latest_step
else:
    step = args.global_step

ckpt_path = os.path.join(model_path, f"global_step_{step}", "actor", "huggingface")
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(ckpt_path)
model.to(device)

# Add the prefix and suffix to the prompt.
prefix = "<\uff5cUser\uff5c>Solve the following math problem step by step. "
suffix = " \nRemenber to put your answer on its own line after \"Answer:\".<\uff5cAssistant\uff5c><think>\n"
prompt = prefix + args.prompt + suffix

# Tokenize the prompt and generate the output.
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
output_ids = model.generate(input_ids, max_length=args.max_length, do_sample=True, temperature=0.2, top_p=1, num_return_sequences=1)
output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated output:")
print(output)