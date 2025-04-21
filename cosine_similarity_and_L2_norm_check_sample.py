#!/usr/bin/env python
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

# Import the model and helper function from train_projection.py
from train_projection import SigLIPProjectionModel, extract_phi_embeddings, CONFIG

# We'll use the AutoTokenizer from transformers to tokenize text prompts.
from transformers import AutoTokenizer

def compute_embedding_metrics(image_path, prompt, device="cuda"):
    """
    Compute the cosine similarity and the L2 norms (magnitudes) of the projected image
    embedding and the text embedding for the given prompt.
    """
    # Initialize the SigLIPProjectionModel (loads both SigLIP and Phi‑3)
    model = SigLIPProjectionModel(
        siglip_model_name=CONFIG["SIGLIP_MODEL"],
        phi_model_name=CONFIG["PHI_MODEL"],
        projection_dim=CONFIG["PROJECTION_DIM"],
        num_layers=CONFIG["PROJECTION_LAYERS"]
    ).to(device)
    model.eval()

    # Preprocess the image.
    transform = transforms.Compose([
        transforms.ToTensor(),  # Assumes image pixels are in [0, 1]
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Shape: [1, C, H, W]

    with torch.no_grad():
        # Get the projected image embedding from the projection network.
        image_embedding = model(image_tensor)  # Shape: [1, embedding_dim]

    # Tokenize the text prompt using the pretrained tokenizer for the Phi‑3 model.
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["PHI_MODEL"])
    encoded = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Extract text embedding using the function from train_projection.py.
    text_embedding = extract_phi_embeddings(model.phi, encoded["input_ids"], encoded["attention_mask"])
    # text_embedding shape: [batch_size, embedding_dim] (here batch_size is 1)

    # Compute cosine similarity between image and text embeddings.
    cosine_sim = F.cosine_similarity(image_embedding, text_embedding, dim=-1).item()

    # Compute the L2 norms (magnitudes) of the embeddings.
    image_norm = torch.norm(image_embedding, p=2).item()
    text_norm = torch.norm(text_embedding, p=2).item()

    return {
        "cosine_similarity": cosine_sim,
        "image_norm": image_norm,
        "text_norm": text_norm
    }

def main():
    parser = argparse.ArgumentParser(description="Cosine Similarity and Magnitude Check between image and text embeddings")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for which to compute text embedding")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = compute_embedding_metrics(args.image, args.prompt, device)
    
    print(f"Cosine Similarity: {metrics['cosine_similarity']:.4f}")
    print(f"Image Embedding Norm: {metrics['image_norm']:.4f}")
    print(f"Text Embedding Norm: {metrics['text_norm']:.4f}")

if __name__ == "__main__":
    main()
