# -*- coding: utf-8 -*-
"""
Generates multimodal conversation dataset using a VLM.
MODIFIED VERSION: Uses STL-10 (default) or CIFAR-10, runs in parallel across GPUs,
and uses the simpler 5-question generation logic from the reference script.
Minimal quality filtering. Saves partial datasets per process.
Corrected processor call to avoid TypeError.
"""

import os
import tempfile
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
# import matplotlib.pyplot as plt # Not needed for parallel run display
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import json
import random
from tqdm import tqdm
import argparse # For command-line arguments
import traceback # For detailed error printing

# --- Configuration ---
# Define the base path on the larger drive for dataset downloads
BASE_DATA_PATH = "/opt/dlami/nvme/ERA-V3-S23-assignment/torchvision_datasets" # MODIFIED
# VLM Model to use
VLM_MODEL_PATH = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# --- Directory Setup ---
# Setup temp dir (optional)
temp_dir = os.path.join(os.path.expanduser("~"), "temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["TMPDIR"] = temp_dir
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir
# Setup HuggingFace cache dir
hf_cache_dir = os.path.join(os.path.expanduser("~"), "hf_cache")
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.makedirs(hf_cache_dir, exist_ok=True)
print(f"HuggingFace cache set to: {hf_cache_dir}")

# --- Dataset Function (Selectable Dataset, Saves to BASE_DATA_PATH) ---
def download_dataset(dataset_name='stl10'):
    """Downloads STL10 or CIFAR10 to BASE_DATA_PATH and returns trainset, classes."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset_root = os.path.join(BASE_DATA_PATH, dataset_name.lower()) # Use BASE_DATA_PATH
    print(f"Using {dataset_name.upper()} dataset from: {dataset_root}")
    os.makedirs(dataset_root, exist_ok=True) # Ensure root exists

    if dataset_name.lower() == 'stl10':
        # Download/Load STL-10 train split
        full_trainset = torchvision.datasets.STL10(root=dataset_root, split='train', download=True, transform=transform)
        classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')
    elif dataset_name.lower() == 'cifar10':
        # Download/Load CIFAR-10 train split
        full_trainset = torchvision.datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'stl10' or 'cifar10'.")
    # Return trainset, not dataloader
    return full_trainset, classes

# --- VLM Loading Function (Loads to specified GPU) ---
def load_vlm_model(gpu_id):
    """Loads VLM model onto specified GPU."""
    device = torch.device(f"cuda:{gpu_id}") # Use specific GPU
    print(f"Loading model '{VLM_MODEL_PATH}' onto {device}...")
    processor = AutoProcessor.from_pretrained(VLM_MODEL_PATH)
    try:
         # Try direct loading to device
         model = AutoModelForImageTextToText.from_pretrained(
             VLM_MODEL_PATH, torch_dtype=torch.float16, device_map={'': device})
    except Exception:
         # Fallback to loading on CPU then moving
         print("Direct device_map failed, loading to CPU then moving.")
         model = AutoModelForImageTextToText.from_pretrained(
             VLM_MODEL_PATH, torch_dtype=torch.float16).to(device)
    model.eval() # Set to evaluation mode
    print(f"Model loaded successfully on {device}.")
    # Return device along with model and processor
    return model, processor, device

# --- Conversation Generation Function (Simpler Logic + Corrected Processor Call) ---
def generate_conversation_data(model, processor, image, class_name, device): # Added device
    """Generate ~5 conversational turns based on the reference script's logic."""
    # Define conversation templates (original simpler version)
    conversation_templates = {
        "conversational": [ "What's happening here?", "Can you describe what's in this picture?", "What do you notice first in this image?", "Is this image interesting or unusual?", "What catches your attention in this scene?", "What stands out the most?", "Can you tell me more about this scene?", "Does this image look familiar to you?" ],
        "detailed_description": [ "What objects are present in the image?", "What are the main colors and shapes in this image?", "Describe the scene in as much detail as possible.", "How is the object positioned in the image?", "What textures or surfaces can you observe?", "Describe the background and the foreground.", "Are there any patterns or repeating elements?", "Is the object in motion or still?" ],
        "complex_reasoning": [ "What could be the function of the object in this image?", "Why might someone be interested in this image?", "What might be happening just outside the frame?", "What could this image tell us about the environment it was taken in?", "What might this object or scene be used for?", "How could someone interact with what's shown here?", "Are there any potential risks or benefits related to this scene?", "What might this image make someone feel, and why?" ]
    }

    conversation = [] # Holds {"human": ..., "assistant": ...} dicts
    selected_questions = []

    # Select one question from each category
    for category, questions in conversation_templates.items():
        if questions: # Check if list is not empty
            selected_questions.append((category, random.choice(questions)))

    # Create a pool of all remaining questions
    all_remaining_questions = []
    selected_q_texts = {q[1] for q in selected_questions}
    for category, questions in conversation_templates.items():
        for question in questions:
            if question not in selected_q_texts:
                 all_remaining_questions.append((category, question))

    # Randomly select 2 more unique questions if possible
    num_needed = 5 - len(selected_questions) # Aim for 5 total turns
    if num_needed > 0 and all_remaining_questions:
        num_to_sample = min(num_needed, len(all_remaining_questions))
        additional_questions = random.sample(all_remaining_questions, num_to_sample)
        selected_questions.extend(additional_questions)

    random.shuffle(selected_questions) # Shuffle the final set of questions

    # Process each selected question
    for category, question in selected_questions:
        try:
            messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
            rgb_image = image.convert("RGB") if image.mode != "RGB" else image # Ensure RGB

            # --- CORRECTED PROCESSOR CALL ---
            inputs = processor.apply_chat_template(
                messages, # Processor finds the image inside messages
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
                # NO explicit images=rgb_image kwarg here
            ).to(device, dtype=torch.float16) # <-- Use passed device
            # --- END CORRECTION ---

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=300,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )

            input_token_len = inputs['input_ids'].shape[1]
            generated_part = generated_ids[:, input_token_len:]
            answer = processor.batch_decode(generated_part, skip_special_tokens=True)[0].strip()

            if answer.startswith("Assistant:"): answer = answer.replace("Assistant:", "").strip()

            if not answer: # Basic check for empty answer
                answer = "[MODEL_RETURNED_EMPTY]"
                tqdm.write(f"Warning: Model returned empty answer for question '{question}'.")

            conversation.append({"human": question, "assistant": answer})

        except Exception as e:
            tqdm.write(f"ERROR generating answer for question '{question}': {e}")
            # Uncomment below for full traceback during debugging
            # tqdm.write(traceback.format_exc())
            conversation.append({
                "human": question,
                "assistant": "[ERROR_DURING_GENERATION]"
            })

    return conversation
# --- End generate_conversation_data ---


# --- Function to save partial dataset ---
def save_partial_dataset(output_dir, data_list, process_id):
    """Saves the generated data list for a specific process."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"dataset_part_{process_id}.json")
    try:
        with open(filename, 'w') as f:
            json.dump(data_list, f, indent=2)
        tqdm.write(f"Saved partial dataset checkpoint for process {process_id} to {filename} ({len(data_list)} items)")
    except Exception as e:
        tqdm.write(f"Error saving partial dataset {filename}: {e}")
# --- End save_partial_dataset ---


# --- Main Execution Logic (Parallelized) ---
def main(args):
    print(f"Loading dataset: {args.dataset_name}...")
    try: trainset, classes = download_dataset(args.dataset_name)
    except ValueError as e: print(e); return
    except Exception as e: print(f"Error loading dataset {args.dataset_name}: {e}"); return

    num_total_images = len(trainset)
    print(f"Total training images in {args.dataset_name}: {num_total_images}"); print(f"Classes: {classes}")

    # --- Determine Index Range for this Process ---
    start_index = args.start_index; end_index = args.end_index
    if start_index < 0: start_index = 0
    if end_index is None or end_index > num_total_images: end_index = num_total_images
    if start_index >= end_index: print(f"Start index {start_index} >= end index {end_index}. No images."); return
    process_range = range(start_index, end_index)
    # ---

    num_images_to_process = len(process_range)
    process_id = f"{args.dataset_name}_gpu{args.gpu_id}_{start_index}-{end_index-1}"
    print(f"Process {process_id} (GPU: {args.gpu_id}) processing indices {start_index} to {end_index-1} ({num_images_to_process} images)")

    # Setup Output Directories
    output_dir = args.output_dir or f"./{args.dataset_name}_vlm_dataset_simple_parallel" # Adjusted default name
    image_output_dir = os.path.join(output_dir, "images"); json_output_dir = os.path.join(output_dir, "json_parts")
    os.makedirs(image_output_dir, exist_ok=True); os.makedirs(json_output_dir, exist_ok=True)
    print(f"Output will be saved in: {output_dir}")

    # Load VLM Model onto assigned GPU
    print(f"Loading VLM model for process {process_id}...")
    try: model, processor, device = load_vlm_model(args.gpu_id)
    except Exception as e: print(f"FATAL: Failed to load model on GPU {args.gpu_id}: {e}"); return
    print(f"Model loaded successfully for process {process_id} on {device}!")

    process_data = []; saved_indices_count = 0
    pbar = tqdm(process_range, desc=f"GPU {args.gpu_id} Processing", unit="image", mininterval=2.0)

    # --- Main Processing Loop (Iterate through assigned indices) ---
    for i in pbar:
        try:
            image_data, label = trainset[i]; class_name = classes[label]
            if isinstance(image_data, torch.Tensor): pil_image = transforms.ToPILImage()(image_data)
            elif isinstance(image_data, Image.Image): pil_image = image_data
            else: tqdm.write(f"Warning: Unexpected type index {i}: {type(image_data)}. Skip."); continue

            pbar.set_description(f"GPU {args.gpu_id} Proc {class_name} ({i})")

            # Generate conversation using the simpler logic
            conversation = generate_conversation_data(model, processor, pil_image, class_name, device) # Pass device

            # --- Save Image and Format Output ---
            if conversation and len(conversation) > 0: # Check if any turns were generated
                if pil_image.mode != 'RGB': pil_image = pil_image.convert('RGB')
                image_filename_base = f"{class_name.replace(' ', '_')}_{i}.png"
                image_save_path = os.path.join(image_output_dir, image_filename_base)
                pil_image.save(image_save_path)
                image_relative_path = os.path.join("images", image_filename_base) # Relative path for JSON

                # Format for LLaVA style
                formatted_llava_style = []
                for turn in conversation: # Include all generated turns
                     human_val = turn.get("human"); assistant_val = turn.get("assistant")
                     if human_val and assistant_val: # Check both keys exist
                           formatted_llava_style.append({"from": "human", "value": human_val})
                           formatted_llava_style.append({"from": "assistant", "value": assistant_val})

                # Add to process data if formatted conversation is not empty
                # No strict filtering applied here, assumes VLM output is usable
                if formatted_llava_style:
                    process_data.append({
                        "image": image_relative_path,
                        "conversations": formatted_llava_style,
                        "original_index": i
                    })

        except KeyboardInterrupt: print(f"\nInterrupted by user on GPU {args.gpu_id}. Saving progress..."); break
        except Exception as e: tqdm.write(f"CRITICAL Error processing image index {i}: {str(e)}\n{traceback.format_exc()}")

        # --- Checkpointing Logic ---
        items_processed_in_run = (i - start_index) + 1
        if items_processed_in_run > 0 and items_processed_in_run % args.save_interval == 0:
             if process_data:
                 checkpoint_id = f"{process_id}_checkpoint_at_{i+1}"
                 save_partial_dataset(json_output_dir, process_data, checkpoint_id)
                 saved_indices_count += len(process_data); process_data = []

    # --- Final Save ---
    if process_data:
        final_id = f"{process_id}_final"
        save_partial_dataset(json_output_dir, process_data, final_id)
        saved_indices_count += len(process_data)

    print(f"Process {process_id} finished. Total conversations saved by this process: {saved_indices_count}")
# --- End main ---


# --- Argument Parser Setup ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conversation dataset (simple version) in parallel using STL10 or CIFAR10.")
    # --- Arguments for Parallelization and Dataset ---
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID (0, 1, 2, 3).")
    parser.add_argument("--start_index", type=int, required=True, help="Starting index (inclusive).")
    parser.add_argument("--end_index", type=int, required=True, help="Ending index (exclusive).")
    parser.add_argument("--dataset_name", type=str, default="stl10", choices=['stl10', 'cifar10'], help="Dataset (stl10 or cifar10). Default: stl10")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (defaults based on dataset name).")
    parser.add_argument("--save_interval", type=int, default=200, help="Checkpoint save interval. Default: 200")

    args = parser.parse_args()

    # Basic validation
    if args.start_index >= args.end_index or args.start_index < 0:
         print(f"Error: Invalid index range --start_index {args.start_index} --end_index {args.end_index}")
    else:
         main(args)
# --- End Script ---