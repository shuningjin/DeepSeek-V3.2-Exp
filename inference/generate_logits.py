"""
Adapted from
https://github.com/AI-Hypercomputer/maxtext/blob/d272058f58671c6ccf3a66cb3e25f49832aa2e40/tests/assets/logits_generation/generate_hf_golden_logits.py#L1

Usage: 

pip install google-cloud-storage
pip install jsonlines

unset NCCL_NET
MP=8 # num gpu
torchrun --nproc-per-node ${MP} \
generate_logits.py \
--ckpt-path <ckpt path> \
--config config_671B_v3.2.json \
--output-path <output path> \
--gcs-bucket <gcs bucket> \
--output-format json \
--prompts 'I love to'
"""

import os
import jsonlines
import pickle
import numpy as np
from argparse import ArgumentParser
from typing import List
import json

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs

from google.cloud import storage


def upload_blob(bucket_name, source_file_name, destination_blob_name):
  """Uploads a file to the bucket."""
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(destination_blob_name)
  blob.upload_from_filename(source_file_name)


@record
def main(
    ckpt_path: str,
    config: str,
    prompts: List[str],
    output_path: str,
    gcs_bucket: str,
    output_format: str = "json",
    force_add_bos: bool = False,
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("load model")
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    print("I'm DeepSeek 👋")

    all_data_to_save = []
    # Process Prompts
    for prompt_text in prompts:
        data_to_save = {"prompt": prompt_text}
        print(f"Processing prompt: {prompt_text}")
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
        if force_add_bos:
          print("concat BOS id")
          bos_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=input_ids.dtype)
          input_ids = torch.cat([bos_ids, input_ids], dim=-1)
        inputs = {"input_ids": input_ids}
        with torch.inference_mode():
          outputs = model.forward(input_ids.to("cuda"), start_pos=0, return_all_logits=True)
        
        # Convert to float32 to maintain precision, move to CPU, convert to numpy
        logits = outputs.cpu().to(torch.float32).numpy()
        
        # Rank 0 collects the data
        if rank == 0:
          # 3. Populate final data dictionary with tensors from inputs and logits
          for key, value in inputs.items():
            new_key = "tokens" if key == "input_ids" else key
            data_to_save[new_key] = value.cpu().numpy()[0]
          data_to_save["logits"] = logits[0]

          print(f"Token length is {len(data_to_save['tokens'])} for prompt: {prompt_text}")
          print(f"raw ids: {data_to_save['tokens']}")

          # 4. Convert numpy arrays to lists if format is json
          if output_format == "json":
            for key, value in data_to_save.items():
              if isinstance(value, np.ndarray):
                data_to_save[key] = value.tolist()

          all_data_to_save.append(data_to_save)

    # Save output strictly on Rank 0
    if rank == 0:
      # 5. Save the collected data
      if output_format == "json":
        with jsonlines.open(output_path, "w") as f:
          f.write_all(all_data_to_save)
      elif output_format == "pickle":
        with open(output_path, "wb") as f:
          pickle.dump(all_data_to_save, f)
      print(f"File is stored locally at {output_path}.")

      if gcs_bucket:
        model_id = "ds32"
        upload_blob(gcs_bucket, output_path, f"golden-logits/{model_id}/{output_path}")
        print(f"File is uploaded to gs://{gcs_bucket}/golden-logits/{model_id}/{output_path}.")

    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


def str2bool(v):
  """Parses a string representation of a boolean value into a Python boolean."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("true"):
    return True
  elif v.lower() in ("false"):
    return False
  else:
    raise argparse.ArgumentTypeError("Boolean value expected (e.g., True or False).")


if __name__ == "__main__":
    parser = ArgumentParser(description="Extract Golden Logits using Native DeepSeek Checkpoints")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path containing tokenizer and sharded safetensors")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--output-path", type=str, required=True, help="File to save outputs to (e.g., logits.jsonl)")
    parser.add_argument("--prompts", type=str, required=False, default="", help="Semicolon separated string of prompts")
    parser.add_argument("--prompt-path", type=str, required=False, default="", help="read prompt from file")
    parser.add_argument("--output-format", type=str, choices=["json", "pickle"], default="json", help="Format to save logits")
    parser.add_argument(
        "--gcs-bucket", type=str, required=False, default=None, help="A GCS bucket to store logits, without gs://."
    )
    parser.add_argument("--force-add-bos", type=str2bool, default=False)

    args = parser.parse_args()
    
    if args.prompts:
      # Split the semicolon-separated string into a list of prompts
      prompt_list = [p.strip() for p in args.prompts.split(";") if p.strip()]
    elif args.prompt_path:
      with open(args.prompt_path, "r") as f:
        prompt = f.read()
        prompt_list = [prompt]
    else:
      raise ValueError("no prompt")
    
    main(
        ckpt_path=args.ckpt_path,
        config=args.config,
        prompts=prompt_list,
        output_path=args.output_path,
        gcs_bucket=args.gcs_bucket,
        output_format=args.output_format,
        force_add_bos=args.force_add_bos,
    )
