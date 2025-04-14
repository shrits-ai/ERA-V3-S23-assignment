# --- Make sure these imports are present ---
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig # Added BitsAndBytesConfig
from peft import PeftModel # Added PeftModel
import argparse
# Ensure SigLIPProjectionModel is available (import or copy class definition)
from train_projection import SigLIPProjectionModel, CONFIG as PROJECTION_CONFIG
# Import QLoRA config for settings
from train_qlora import CONFIG as QLORA_CONFIG
# ---

# --- REMOVE the old load_projection_model function ---

# --- ADD this new loading function ---
def load_models_for_inference(phi_model_name, siglip_model_name, projection_checkpoint_path, adapter_checkpoint_path, projection_dim, projection_layers, quant_config):
    """Loads the base Phi model, applies the QLoRA adapter, and loads the projection model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = QLORA_CONFIG["COMPUTE_DTYPE"] # Get compute_dtype from QLoRA config

    # 1. Load the base Phi-3 model with quantization
    print(f"Loading base Phi-3 model: {phi_model_name} with quantization...")
    phi_model = AutoModelForCausalLM.from_pretrained(
        phi_model_name,
        quantization_config=quant_config,
        torch_dtype=compute_dtype, # Use compute_dtype
        device_map="auto",
        trust_remote_code=True # Add if necessary for Phi-3
    )
    # Ensure pad token is set if needed (copy from train_qlora.py MultimodalPhiModel init)
    phi_tokenizer = AutoTokenizer.from_pretrained(phi_model_name, trust_remote_code=True)
    if phi_tokenizer.pad_token is None:
        print("Setting pad_token to eos_token for tokenizer.")
        phi_tokenizer.pad_token = phi_tokenizer.eos_token
        phi_model.config.pad_token_id = phi_tokenizer.pad_token_id


    # 2. Load the QLoRA adapter
    print(f"Loading QLoRA adapter from: {adapter_checkpoint_path}...")
    # Use PeftModel to load the adapter onto the base model
    # adapter_checkpoint_path should be the DIRECTORY (e.g., './phi3_qlora_adapter/final_model/')
    model_with_adapter = PeftModel.from_pretrained(phi_model, adapter_checkpoint_path)
    model_with_adapter.eval() # Set the combined model to evaluation mode
    print("Phi-3 model with QLoRA adapter loaded.")


    # 3. Load the SigLIP projection model
    print(f"Loading SigLIP projection model from: {projection_checkpoint_path}...")
    projection_model = SigLIPProjectionModel(
        siglip_model_name=siglip_model_name,
        phi_model_name=phi_model_name, # Needed for internal dims, but LLM part won't be used
        projection_dim=projection_dim,
        num_layers=projection_layers # Use num_layers from train_projection
    )
    # Load the projection layer's weights ONLY
    # projection_checkpoint_path should point to the .pt file from train_projection
    proj_checkpoint = torch.load(projection_checkpoint_path, map_location="cpu")
    projection_model.projection.load_state_dict(proj_checkpoint["model_state_dict"])
    projection_model.to(device, dtype=compute_dtype).eval() # Move to device and set to eval
    print("SigLIP projection model loaded.")


    return model_with_adapter, projection_model, phi_tokenizer

# --- Keep process_image function as is ---
# --- Ensure this function definition is present in your inference.py ---
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

    # Apply transform (Use SigLIP's expected transform)
    if transform is None:
        # Using mean/std [0.5, 0.5, 0.5] is common for CLIP-like models if specific ones aren't known
        # Or use the CIFAR ones if you trained the projection layer that way, but SigLIP's own preprocessing is ideal.
        # Let's stick to a standard [0.5, 0.5, 0.5] normalization often used with vision transformers.
        transform = transforms.Compose([
            transforms.Resize((384, 384)),  # SigLIP default input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]) # Common normalization
        ])

    # Transform image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    return image_tensor
# --- End of process_image function definition ---

def generate_text_from_image(phi_model_with_adapter, projection_model, image_tensor, phi_tokenizer, device="cuda", max_length=100, prompt="Describe what you see in this image."):
    """
    Generate text from an image using the PEFT adapter model and projection model.
    (Modified function signature and internal logic)
    """
    # Move image to the projection model's device and dtype
    proj_layer = projection_model.projection[0] if isinstance(projection_model.projection, nn.Sequential) else projection_model.projection
    image_tensor = image_tensor.to(next(projection_model.parameters()).device, dtype=proj_layer.weight.dtype) # Handle Sequential case


    # Set models to evaluation mode
    phi_model_with_adapter.eval()
    projection_model.eval()


    with torch.no_grad():
        # 1. Get projected image embeddings from the SigLIP projection model
        image_embeddings = projection_model(image_tensor) # Shape: [batch_size, proj_dim]


        # 2. Get Phi's input embedding layer
        embedding_layer = phi_model_with_adapter.get_input_embeddings()
        phi_embedding_dim = embedding_layer.weight.shape[-1]


        # Ensure projection dim matches Phi embedding dim
        if image_embeddings.shape[-1] != phi_embedding_dim:
             raise ValueError(f"Projected image embedding dimension ({image_embeddings.shape[-1]}) "
                              f"does not match Phi-3 embedding dimension ({phi_embedding_dim}).")


        # Prepare image embeddings for concatenation: ensure same device and dtype as input_embeddings layer
        image_embeddings = image_embeddings.to(embedding_layer.weight.device, dtype=embedding_layer.weight.dtype)
        image_embeddings_unsqueezed = image_embeddings.unsqueeze(1) # [batch_size, 1, phi_emb_dim]


        # 3. Tokenize the prompt and get its embeddings
        prompt_tokens = phi_tokenizer(prompt, return_tensors="pt").to(embedding_layer.weight.device)
        prompt_embeddings = embedding_layer(prompt_tokens.input_ids) # [batch_size, prompt_len, phi_emb_dim]


        # 4. Combine embeddings: [ImageEmb, PromptEmb] -> Phi starts generation from here
        combined_embeddings = torch.cat([image_embeddings_unsqueezed, prompt_embeddings], dim=1)
        # Shape: [batch_size, 1+prompt_len, phi_emb_dim]
        batch_size = combined_embeddings.shape[0]
        sequence_length = combined_embeddings.shape[1]


        # 5. Create Attention Mask
        attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.long, device=combined_embeddings.device)


        # 6. Create dummy input IDs corresponding to the embeddings
        # Use UNK token ID for the image part, then the actual prompt token IDs
        unk_token_id = phi_tokenizer.unk_token_id if phi_tokenizer.unk_token_id is not None else phi_tokenizer.bos_token_id
        
        # Create input_ids with UNK for image and actual tokens for prompt
        input_ids = torch.cat([
            torch.tensor([[unk_token_id]], device=prompt_tokens.input_ids.device),
            prompt_tokens.input_ids
        ], dim=1)


        # 7. Generate text passing BOTH input_ids and inputs_embeds
        outputs = phi_model_with_adapter.generate(
            input_ids=input_ids,
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
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


# --- Make sure the main function uses the new loading function and arguments ---
def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = QLORA_CONFIG["COMPUTE_DTYPE"] # Get compute_dtype

    # Define Quantization Config (using settings from QLoRA training)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=QLORA_CONFIG["QUANT_TYPE"], # Use quant_type from QLoRA config
        bnb_4bit_use_double_quant=True
    )

    # Load models and tokenizer using the NEW function and argument names
    phi_model_with_adapter, projection_model, phi_tokenizer = load_models_for_inference(
        phi_model_name=args.phi_model_name,
        siglip_model_name=args.siglip_model_name,
        projection_checkpoint_path=args.projection_model_path, # Use new argument
        adapter_checkpoint_path=args.adapter_checkpoint_path,   # Use new argument
        projection_dim=args.projection_dim,
        projection_layers=args.projection_layers,
        quant_config=quantization_config
    )

    # Process image
    image_tensor = process_image(args.image_path)

    # Generate text
    generated_text = generate_text_from_image(
        phi_model_with_adapter=phi_model_with_adapter,
        projection_model=projection_model,
        image_tensor=image_tensor,
        phi_tokenizer=phi_tokenizer,
        device=device,
        max_length=args.max_length,
        prompt=args.prompt
    )

    # Print results
    print(f"Generated text for image {args.image_path}:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

# --- Make sure the Argument Parser uses the new argument names ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with SigLIP to Phi-3 projection model")

    # Model arguments
    parser.add_argument("--siglip_model_name", type=str, default="google/siglip-so400m-patch14-384",
                        help="Name of the SigLIP model")
    parser.add_argument("--phi_model_name", type=str, default="microsoft/phi-3-mini-4k-instruct",
                        help="Name of the Phi-3 model")
    # --- Use the NEW Checkpoint Arguments ---
    parser.add_argument("--projection_model_path", type=str, required=True,
                        help="Path to the trained projection model checkpoint (.pt file from train_projection.py)")
    parser.add_argument("--adapter_checkpoint_path", type=str, required=True,
                        help="Path to the trained QLoRA adapter directory (e.g., ./phi3_qlora_adapter/final_model/)")
    # --- ---
    parser.add_argument("--projection_dim", type=int, default=PROJECTION_CONFIG["PROJECTION_DIM"], # Use dim from projection config
                        help="Dimension of the projection layer")
    parser.add_argument("--projection_layers", type=int, default=PROJECTION_CONFIG["PROJECTION_LAYERS"], # Use layers from projection config
                        help="Number of layers in projection MLP")

    # Inference arguments
    parser.add_argument("--image_path", type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--prompt", type=str, default="What object is shown in this image? Describe it in detail.",
                        help="Prompt to guide the image description generation")

    args = parser.parse_args()

    # Make sure PROJECTION_CONFIG and QLORA_CONFIG are available
    try:
        from train_projection import CONFIG as PROJECTION_CONFIG
        from train_qlora import CONFIG as QLORA_CONFIG
    except ImportError:
        print("Warning: Could not import configs from training scripts. Using defaults.")
        # Define fallbacks if necessary
        PROJECTION_CONFIG = {"PROJECTION_DIM": 2048, "PROJECTION_LAYERS": 2}
        QLORA_CONFIG = {"COMPUTE_DTYPE": torch.bfloat16, "QUANT_TYPE": "nf4"} # Example defaults

    main(args)