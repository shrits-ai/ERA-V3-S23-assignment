import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer
import argparse
from train_projection import SigLIPProjectionModel

def load_projection_model(siglip_model_name, phi_model_name, checkpoint_path, projection_dim=2048):
    """
    Load the trained projection model.
    
    Args:
        siglip_model_name (str): Name of the SigLIP model
        phi_model_name (str): Name of the Phi-3 model
        checkpoint_path (str): Path to the trained projection model checkpoint
        projection_dim (int): Dimension of the projection layer
        
    Returns:
        SigLIPProjectionModel: Loaded model
    """
    # Create model
    model = SigLIPProjectionModel(
        siglip_model_name=siglip_model_name,
        phi_model_name=phi_model_name,
        projection_dim=projection_dim
    )
    
    # Load trained projection weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.projection.load_state_dict(checkpoint["model_state_dict"])
    
    return model

def process_image(image_path, transform=None):
    """
    Process an image for inference.
    
    Args:
        image_path (str): Path to the image
        transform: Optional transform to apply
        
    Returns:
        torch.Tensor: Processed image tensor
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Apply transform
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # SigLIP uses 384x384 images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    # Transform image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def generate_text_from_image(model, image_tensor, phi_tokenizer, device="cuda", max_length=100):
    """
    Generate text from an image using the projection model.
    
    Args:
        model (SigLIPProjectionModel): Trained projection model
        image_tensor (torch.Tensor): Processed image tensor
        phi_tokenizer: Phi-3 tokenizer
        device (str): Device to use for inference
        max_length (int): Maximum length of generated text
        
    Returns:
        str: Generated text
    """
    # Move model and image to device
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Get projected embeddings
        projected_embeddings = model(image_tensor)
        
        # Get embedding layer from Phi model
        embedding_layer = model.phi.get_input_embeddings()
        
        # Create input embeddings for Phi model
        # We'll use the projected embedding as the first token embedding
        # and add a BOS token embedding
        bos_token_id = phi_tokenizer.bos_token_id
        bos_embedding = embedding_layer(torch.tensor([[bos_token_id]], device=device))
        
        # Combine embeddings: [batch_size, 1 + 1, hidden_size]
        # Reshape projected embeddings to match expected shape
        projected_embeddings = projected_embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        combined_embeddings = torch.cat([bos_embedding, projected_embeddings], dim=1)
        
        # Generate text using the Phi model with the combined embeddings
        outputs = model.phi.generate(
            inputs_embeds=combined_embeddings,
            max_length=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=phi_tokenizer.pad_token_id,
            eos_token_id=phi_tokenizer.eos_token_id
        )
        
        # Decode the generated text
        generated_text = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    return generated_text

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Phi tokenizer
    phi_tokenizer = AutoTokenizer.from_pretrained(args.phi_model_name)
    
    # Load projection model
    model = load_projection_model(
        siglip_model_name=args.siglip_model_name,
        phi_model_name=args.phi_model_name,
        checkpoint_path=args.checkpoint_path,
        projection_dim=args.projection_dim
    )
    
    # Process image
    image_tensor = process_image(args.image_path)
    
    # Generate text
    generated_text = generate_text_from_image(
        model=model,
        image_tensor=image_tensor,
        phi_tokenizer=phi_tokenizer,
        device=device,
        max_length=args.max_length
    )
    
    # Print results
    print(f"Generated text for image {args.image_path}:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with SigLIP to Phi-3 projection model")
    
    # Model arguments
    parser.add_argument("--siglip_model_name", type=str, default="google/siglip-so400m-patch14-384",
                        help="Name of the SigLIP model")
    parser.add_argument("--phi_model_name", type=str, default="microsoft/phi-3-mini-4k-instruct",
                        help="Name of the Phi-3 model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the trained projection model checkpoint")
    parser.add_argument("--projection_dim", type=int, default=2048,
                        help="Dimension of the projection layer")
    
    # Inference arguments
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    
    args = parser.parse_args()
    main(args) 