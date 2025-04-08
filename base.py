import os
import tempfile
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForVision2Seq, CLIPImageProcessor
import numpy as np

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

def load_external_image(image_path="owl.png"):
    """Load an external image file."""
    try:
        # Load the image using PIL
        image = Image.open(image_path)
        
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        print(f"Successfully loaded image: {image_path}")
        print(f"Image size: {image.size}, Mode: {image.mode}")
        
        # Save a copy for debugging
        image.save("loaded_image.png")
        
        return image, "external image"  # Return the image and a placeholder label
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # If there's an error, fall back to CIFAR-10
        print("Falling back to CIFAR-10 image")
        trainloader, _ = download_cifar10()
        return load_single_image(trainloader)

def load_vlm_model():
    """Load the BLIP-2 model for better image captioning."""
    # Use BLIP-2 with OPT-2.7B
    model_path = "Salesforce/blip2-opt-2.7b"
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load the model with float16 precision to save memory
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
    
    # Move to GPU
    model = model.to("cuda")
    
    return model, processor

def generate_image_descriptions(model, processor, image, num_descriptions=1):
    """Generate a description about the image using BLIP-2."""
    # Save the image being passed to the model for debugging
    image.save("model_input_image.png")
    
    # Process the image and generate a caption
    inputs = processor(images=image, return_tensors="pt").to("cuda")
    
    # Generate with the model
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # Decode the output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return [generated_text]  # Return as a list to maintain compatibility

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
        print(f"{i}. {desc}")
        print("-" * 50)

def main():
    # Download CIFAR-10 and get a dataloader
    trainloader, classes = download_cifar10()
    
    # Load a single image from CIFAR-10
    image, label = load_single_image(trainloader)
    
    # Get the class name
    class_name = classes[label]
    
    # Load the VLM model
    model, processor = load_vlm_model()
    
    # Generate descriptions
    descriptions = generate_image_descriptions(model, processor, image)
    
    # Display results
    display_image_with_descriptions(image, class_name, descriptions)

if __name__ == "__main__":
    main()
