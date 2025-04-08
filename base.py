import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForImageTextToText
import numpy as np

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
    
    return pil_image, labels[0].item()

def load_vlm_model():
    """Load the SmolVLM2 model from HuggingFace."""
    model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Load the model with float32 precision instead of bfloat16
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # Changed from bfloat16 to float32
    )
    
    # Then explicitly move it to GPU
    model = model.to("cuda")
    
    # Set the attention implementation
    model._attn_implementation = "flash_attention_2"
    
    return model, processor

def generate_image_descriptions(model, processor, image, num_descriptions=5):
    """Generate multiple descriptions about the image using SmolVLM2."""
    descriptions = []
    
    # Different prompts to get varied descriptions
    prompts = [
        "Describe this image in detail.",
        "What objects can you see in this image?",
        "What colors are prominent in this image?",
        "Describe the composition of this image.",
        "What is the main subject of this image?",
        "Is there any action happening in this image?",
        "What is the setting or background of this image?",
        "Describe the texture and patterns visible in this image.",
        "What emotions does this image evoke?",
        "Describe this image as if to someone who cannot see it."
    ]
    
    # Use different prompts to get varied descriptions
    for i in range(min(num_descriptions, len(prompts))):
        # Debug: Print image information to verify it's being passed correctly
        print(f"Processing image: {type(image)}, Size: {image.size}, Mode: {image.mode}")
        
        # Process the image and text separately
        vision_inputs = processor.image_processor(images=image, return_tensors="pt")
        text_inputs = processor.tokenizer(prompts[i], return_tensors="pt")
        
        # Verify the image tensor shape
        print(f"Image tensor shape: {vision_inputs.pixel_values.shape}")
        
        # Combine them manually
        inputs = {
            "pixel_values": vision_inputs.pixel_values.to("cuda"),
            "input_ids": text_inputs.input_ids.to("cuda"),
            "attention_mask": text_inputs.attention_mask.to("cuda")
        }
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        generated_text = processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        descriptions.append(generated_text)
    
    return descriptions

def display_image_with_descriptions(image, class_name, descriptions, save_path="output_image.png"):
    """Display the image and its descriptions, with options for headless environments."""
    # Save the image to a file
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(f"CIFAR-10 Class: {class_name}")
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()  # Close the figure to free memory
    
    print(f"Image saved to {save_path}")
    print(f"Image class: {class_name}")
    print("\nSmolVLM2 Descriptions:")
    for i, desc in enumerate(descriptions, 1):
        print(f"{i}. {desc}")
        print("-" * 50)

def main():
    # Download CIFAR-10 and get a dataloader
    trainloader, classes = download_cifar10()
    
    # Load a single image
    image, label = load_single_image(trainloader)
    class_name = classes[label]
    
    # Load the VLM model
    model, processor = load_vlm_model()
    
    # Generate descriptions
    descriptions = generate_image_descriptions(model, processor, image, num_descriptions=7)
    
    # Display results
    display_image_with_descriptions(image, class_name, descriptions)

if __name__ == "__main__":
    main()
