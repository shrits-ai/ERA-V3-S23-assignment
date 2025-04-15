import os
import tempfile
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np
import json
import random
from tqdm import tqdm

# Create a temporary directory in the user's home directory
temp_dir = os.path.join(os.path.expanduser("~"), "temp")
os.makedirs(temp_dir, exist_ok=True)
os.environ["TMPDIR"] = temp_dir
os.environ["TEMP"] = temp_dir
os.environ["TMP"] = temp_dir

# Set the cache directory for Hugging Face to a location with sufficient space
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.expanduser("~"), "hf_cache")
os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)

def download_cifar10():
    """Download CIFAR-10 dataset and return the train loader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Download and load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=1,
        shuffle=True
    )
    
    # Class names for CIFAR-10
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, classes

def load_single_image(trainloader):
    """Load a single image from the CIFAR-10 dataset."""
    # Get a single batch
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    
    # Convert tensor to PIL Image
    image = images[0].permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    
    print(f"Image size: {pil_image.size}")
    
    # Save the original image for debugging
    pil_image.save("original_image.png")
    
    return pil_image, labels[0].item()

def load_vlm_model():
    """Load SmolVLM 2 model for detailed image descriptions."""
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Using float16 instead of bfloat16 for wider compatibility
    ).to("cuda")
    
    return model, processor

import random
import torch

def generate_conversation_data(model, processor, image, class_name):
    """
    Generate conversational data with a robust fallback for the first question only.
    
    The conversation includes:
      1) A default anchor question that enforces the CIFAR‑10 label with fallback if uncertain.
      2) 5 additional questions (3, one from each category; and 2 extra chosen overall)
         without any fallback mechanism.
    """
    image.save("model_input_image.png")
    
    # Known CIFAR-10 classes for reference.
    cifar10_classes = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
    
    # Prompt pools: each category has prompts with and without label.
    conversation_templates = {
        "class_identification": {
            "with_label": [
                f"Is this a {class_name}? Describe its features.",
                f"This image shows a {class_name}. Can you describe its characteristics?",
                f"What are the distinctive features of this {class_name}?",
                f"How would you recognize a {class_name} from its appearance?",
                f"Identify this {class_name} and list its main traits.",
                f"What details confirm this is a {class_name}?"
            ],
            "without_label": [
                "What object is shown in this image? Describe its features.",
                "Can you describe the main characteristics of the object in this image?",
                "What distinctive features can you identify in this image?",
                "How would you recognize the object based on its appearance?"
            ]
        },
        "detailed_description": {
            "with_label": [
                f"What does this {class_name} look like?",
                f"Describe the appearance of this {class_name}.",
                f"What are the main colors and shapes of this {class_name}?",
                f"How is this {class_name} positioned in the image?",
                f"What textures or patterns do you observe on this {class_name}?",
                f"Give a detailed visual description of this {class_name}."
            ],
            "without_label": [
                "What does the object in the image look like?",
                "Describe the appearance of the object in this image.",
                "What are the main colors and shapes in the image?",
                "What textures or patterns can you see in this image?"
            ]
        },
        "complex_reasoning": {
            "with_label": [
                f"What is the typical function or purpose of a {class_name}?",
                f"How do people interact with a {class_name}?",
                f"What makes this {class_name} stand out compared to similar objects?",
                f"In what environments would you usually find a {class_name}?",
                f"What interesting facts do you know about {class_name}s?",
                f"Why is a {class_name} considered important in its context?"
            ],
            "without_label": [
                "What is the function or purpose of the object shown?",
                "How might people typically interact with the object in this image?",
                "What makes the object in this image unique?",
                "In what settings would you commonly find this kind of object?"
            ]
        }
    }
    
    # Merge the with_label and without_label prompts into one list for each category.
    for category in conversation_templates:
        with_label = conversation_templates[category]["with_label"]
        without_label = conversation_templates[category]["without_label"]
        conversation_templates[category] = with_label + without_label  # Total 10 prompts per category.
    
    # Build the overall pool of prompts from all categories.
    overall_pool = []
    for cat in conversation_templates:
        overall_pool.extend([(cat, q) for q in conversation_templates[cat]])
    
    conversation = []
    
    # Default anchor question with fallback.
    default_first_question = (
        f"Based on the CIFAR‑10 label, this image should depict a {class_name}. "
        f"Could you confirm this and describe its features?"
    )
    
    def is_uncertain_or_mismatch(answer_text, true_label):
        """Check if the answer indicates uncertainty or references a different class."""
        lower = answer_text.lower()
        dynamic_keyword = f"it is not {true_label}".lower()
        uncertain_keywords = [
            "not sure", "unclear", "can't tell", "cannot tell",
            "i'm sorry", "i cannot", "i can't", dynamic_keyword
        ]
        if any(kw in lower for kw in uncertain_keywords):
            return True
        for c in cifar10_classes:
            if c != true_label and c in lower:
                return True
        return False
    
    def fallback_response(label):
        """Return a hard-coded fallback response that confirms the correct class label."""
        return f"Yes, this is a {label}. It shows the typical characteristics of a {label}."
    
    def ask_question(question_text, use_fallback=True):
        """Send a prompt and return the answer.
           If use_fallback is True, check for uncertainty and apply fallback.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_text},
                ]
            },
        ]
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.float16)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=300
                )
            
            raw_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            if "User:" in raw_answer:
                raw_answer = raw_answer.split("User:")[-1].strip()
            if "Assistant:" in raw_answer:
                raw_answer = raw_answer.split("Assistant:")[-1].strip()
            
            if use_fallback and is_uncertain_or_mismatch(raw_answer, class_name):
                return fallback_response(class_name)
            else:
                return raw_answer
            
        except Exception:
            return fallback_response(class_name) if use_fallback else ""
    
    # 1) Ask the first (anchor) question with fallback.
    first_answer = ask_question(default_first_question, use_fallback=True)
    conversation.append({
        "human": default_first_question,
        "assistant": first_answer
    })
    
    # 2) Select 3 random questions (one per category).
    selected_questions = []
    for category in conversation_templates:
        q = random.choice(conversation_templates[category])
        selected_questions.append((category, q))
    
    # 3) Select 2 additional random questions from the overall pool (avoiding duplicates).
    already_selected = set(q for _, q in selected_questions)
    extra_candidates = [entry for entry in overall_pool if entry[1] not in already_selected]
    extra_questions = random.sample(extra_candidates, 2) if len(extra_candidates) >= 2 else extra_candidates
    
    # Merge the additional questions to have 5 in total.
    combined_questions = selected_questions + extra_questions
    random.shuffle(combined_questions)
    
    # 4) Process each additional question WITHOUT fallback.
    for category, question in combined_questions:
        answer = ask_question(question, use_fallback=False)
        conversation.append({
            "human": question,
            "assistant": answer
        })
    
    return conversation




def save_conversation_to_file(image, class_name, conversation, index, output_dir="llava_dataset"):
    """Save the conversation data to a file in a format similar to LLaVA-Instruct-150k."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the image
    image_filename = f"{output_dir}/image_{index}_{class_name}.png"
    image.save(image_filename)
    
    # Format the conversation data
    conversation_data = {
        "image": image_filename,
        "class": class_name,
        "conversations": conversation
    }
    
    # Save as JSON
    json_filename = f"{output_dir}/conversation_{index}_{class_name}.json"
    with open(json_filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)
    
    return json_filename

def display_image_with_descriptions(image, class_name, descriptions, save_path="output_image.png"):
    """Display the image and its descriptions, with options for headless environments."""
    # Save the image to a file
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"Class: {class_name}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
    
    print(f"Image saved to {save_path}")
    print(f"Image class: {class_name}")
    print("\nImage Descriptions:")
    for i, desc in enumerate(descriptions, 1):
        if isinstance(desc, dict):
            # If it's a dictionary with human/assistant keys
            print(f"{i}. Human: {desc.get('human', '')}")
            print(f"   Assistant: {desc.get('assistant', '')}")
        else:
            # If it's just a string
            print(f"{i}. {desc}")
        print("-" * 50)

def save_dataset(output_dir, data_list, split_ratio=0.9):
    """Save the dataset in a format suitable for training."""
    # Create directories
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    
    # Shuffle data
    random.shuffle(data_list)
    
    # Split into train/val
    split_idx = int(len(data_list) * split_ratio)
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]
    
    # Save train and val JSON files
    with open(os.path.join(output_dir, "train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, "val.json"), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    # Create metadata
    metadata = {
        "dataset_name": "CIFAR10-VLM",
        "num_images": len(data_list),
        "num_train": len(train_data),
        "num_val": len(val_data),
        "classes": ["airplane", "automobile", "bird", "cat", "deer", 
                   "dog", "frog", "horse", "ship", "truck"]
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset saved to {output_dir}")
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")

def main():
    # Download CIFAR-10 and get a dataloader
    print("Downloading CIFAR-10 dataset...")
    trainloader, classes = download_cifar10()
    
    # Create output directory
    output_dir = "cifar10_vlm_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the VLM model
    print("Loading SmolVLM 2 model...")
    model, processor = load_vlm_model()
    print("Model loaded successfully!")
    
    # Store all data entries
    all_data = []
    
    # Process CIFAR-10 images
    num_images = 100  # Aim for 1000+ images for a decent dataset
    
    # Create a tqdm progress bar
    pbar = tqdm(range(num_images), desc="Processing CIFAR-10 images", unit="image")
    
    for i in pbar:
        try:
            # Get a single image
            image, label = load_single_image(trainloader)
            class_name = classes[label]
            
            # Update progress bar description with current class
            pbar.set_description(f"Processing {class_name} ({i+1}/{num_images})")
            
            # Generate conversation data (silently)
            conversation = generate_conversation_data(model, processor, image, class_name)
            
            # Save image
            image_filename = f"images/{class_name}_{i}.png"
            image_path = os.path.join(output_dir, image_filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image.save(image_path)
            
            # Format for training
            formatted_conversation = []
            for entry in conversation:
                formatted_conversation.append({"from": "human", "value": entry["human"]})
                formatted_conversation.append({"from": "assistant", "value": entry["assistant"]})
            
            # Add to dataset
            all_data.append({
                "image": image_filename,
                "conversations": formatted_conversation
            })
            
            # Save progress every 10 images for testing, or 100 for production
            if (i+1) % 10 == 0:
                save_dataset(output_dir, all_data)
                
        except Exception as e:
            # Log errors but continue processing
            pbar.write(f"Error processing image {i}: {str(e)}")
    
    # Final save
    save_dataset(output_dir, all_data)
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
