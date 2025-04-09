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

def generate_conversation_data(model, processor, image, class_name):
    """Generate conversational data similar to LLaVA-Instruct-150k for the given image."""
    # Save the image being passed to the model for debugging
    image.save("model_input_image.png")
    
    # Define conversation templates based on image type/class
    conversation_templates = {
        # Category A - Conversational
        "conversational": [
            "What's happening here?",
            "Can you describe what's in this picture?",
            "What do you notice first in this image?",
            "Is this image interesting or unusual?",
            "What catches your attention in this scene?",
            "What stands out the most?",
            "Can you tell me more about this scene?",
            "Does this image look familiar to you?"
        ],
        # Category B - Detailed Description
        "detailed_description": [
            "What objects are present in the image?",
            "What are the main colors and shapes in this image?",
            "Describe the scene in as much detail as possible.",
            "How is the object positioned in the image?",
            "What textures or surfaces can you observe?",
            "Describe the background and the foreground.",
            "Are there any patterns or repeating elements?",
            "Is the object in motion or still?"
        ],
        # Category C - Complex Reasoning
        "complex_reasoning": [
            "What could be the function of the object in this image?",
            "Why might someone be interested in this image?",
            "What might be happening just outside the frame?",
            "What could this image tell us about the environment it was taken in?",
            "What might this object or scene be used for?",
            "How could someone interact with what's shown here?",
            "Are there any potential risks or benefits related to this scene?",
            "What might this image make someone feel, and why?"
        ]
    }
    
    # Create a conversation structure
    conversation = []
    
    # Select questions for this image
    selected_questions = []
    
    # First, select one question from each category
    for category, questions in conversation_templates.items():
        selected_questions.append((category, random.choice(questions)))
    
    # Create a pool of all remaining questions
    all_remaining_questions = []
    for category, questions in conversation_templates.items():
        for question in questions:
            if not any(q[1] == question for q in selected_questions):
                all_remaining_questions.append((category, question))
    
    # Randomly select 2 more questions from the remaining pool
    additional_questions = random.sample(all_remaining_questions, 2)
    selected_questions.extend(additional_questions)
    
    # Shuffle the questions to mix up the order
    random.shuffle(selected_questions)
    
    # Process each selected question
    for category, question in selected_questions:
        try:
            # Format messages for SmolVLM 2
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ]
                },
            ]
            
            # Apply chat template
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.float16)
            
            # Generate with the model
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=300
                )
            
            # Decode the output
            answer = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            
            # Clean up the answer - remove any "User:" or "Assistant:" prefixes
            if "User:" in answer:
                answer = answer.split("User:")[-1].strip()
            if "Assistant:" in answer:
                answer = answer.split("Assistant:")[-1].strip()
            
            # Add to conversation
            conversation.append({
                "human": question,
                "assistant": answer
            })
            
        except Exception as e:
            # Silently handle errors to avoid cluttering the output
            conversation.append({
                "human": question,
                "assistant": f"I apologize, but I couldn't analyze this image properly."
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
    num_images = 11  # Aim for 1000+ images for a decent dataset
    
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
