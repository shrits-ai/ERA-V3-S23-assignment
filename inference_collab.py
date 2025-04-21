#!/usr/bin/env python
import os
import torch
# import torch.nn.functional as F # Not explicitly used in generate
import torchvision.transforms as transforms
from PIL import Image
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Attempt to import necessary components from your training scripts
# Ensure these scripts are in the same directory or accessible in the Python path
try:
    from train_projection import SigLIPProjectionModel, CONFIG as PROJECTION_CONFIG
    from train_qlora import CONFIG as QLORA_CONFIG # For compute_dtype, quant_type
except ImportError:
    print("Warning: Could not import from train_projection.py or train_qlora.py.")
    print("Falling back to default/example configurations. Ensure these match your trained models.")
    # Define fallback configs if imports fail - ADJUST THESE CAREFULLY
    PROJECTION_CONFIG = {"PROJECTION_DIM": 4096, "PROJECTION_LAYERS": 2, "PHI_MODEL": "microsoft/phi-3-mini-4k-instruct"} # Example
    QLORA_CONFIG = {"COMPUTE_DTYPE": torch.bfloat16, "QUANT_TYPE": "nf4"} # Example

# --- DynamicCache Patch specific to Phi-3 ---
# Keep this as it might be necessary for older transformers versions with Phi-3
def patch_dynamic_cache():
    """Applies a patch for DynamicCache if needed for Phi-3 generation."""
    try:
        from transformers.models.phi3.modeling_phi3 import DynamicCache
        if not hasattr(DynamicCache, "get_max_length"):
            print("Monkey patching DynamicCache: aliasing get_max_length to get_seq_length")
            DynamicCache.get_max_length = DynamicCache.get_seq_length
    except ImportError:
         print("Could not import DynamicCache from transformers.models.phi3.modeling_phi3. Skipping patch.")
    except Exception as e:
        print(f"Warning: Failed to patch DynamicCache: {e}")
# --- End DynamicCache Patch ---

def load_models_for_inference(
    phi3_model_name, # Changed param name for clarity
    siglip_model_name,
    projection_checkpoint_path,
    adapter_checkpoint_path,
    projection_dim, # From PROJECTION_CONFIG
    projection_layers, # From PROJECTION_CONFIG
    quant_config
    ):
    """Loads the base model, adapter, projection model, and tokenizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = QLORA_CONFIG.get("COMPUTE_DTYPE", torch.bfloat16) # Use .get for safety if import failed

    print(f"Loading base Phi-3 model: {phi3_model_name} with quantization...")
    # Use device_map="auto" for automatic placement of the quantized model
    # The ValueError might still occur here if library versions are incompatible
    phi_model = AutoModelForCausalLM.from_pretrained(
        phi3_model_name,
        quantization_config=quant_config,
        torch_dtype=compute_dtype,
        device_map="auto", # <= This handles device placement for the quantized model
        trust_remote_code=True, # Often needed for Phi models
        attn_implementation="eager" # <<< Add this
    )
    print("Base Phi-3 model loaded.")

    print(f"Loading tokenizer for {phi3_model_name}...")
    phi_tokenizer = AutoTokenizer.from_pretrained(phi3_model_name, trust_remote_code=True)
    # Set pad token if missing (common for Phi models)
    if phi_tokenizer.pad_token is None:
        print("Setting pad_token to eos_token for tokenizer.")
        phi_tokenizer.pad_token = phi_tokenizer.eos_token
        # Ensure the model's config also uses this pad token ID
        # This should happen BEFORE loading the adapter
        phi_model.config.pad_token_id = phi_tokenizer.pad_token_id
    print(f"Tokenizer loaded. Pad token ID: {phi_tokenizer.pad_token_id}")

    print(f"Loading QLoRA adapter trained for Phi-3 from: {adapter_checkpoint_path}...")
    # Load the PEFT adapter onto the quantized base model
    # Ensure the adapter was trained for the specific phi3_model_name being loaded
    model_with_adapter = PeftModel.from_pretrained(phi_model, adapter_checkpoint_path)
    model_with_adapter.eval() # Set adapter model to evaluation mode
    print("Phi-3 model with QLoRA adapter loaded.")

    print(f"Loading SigLIP projection model (trained for Phi-3)...")
    # Instantiate projection model using definition from train_projection.py
    # Pass the expected dimensions from PROJECTION_CONFIG
    projection_model = SigLIPProjectionModel(
        siglip_model_name=siglip_model_name,
        phi_model_name=phi3_model_name, # Pass base model name
        projection_dim=projection_dim,
        num_layers=projection_layers
    )
    print(f"Loading projection checkpoint from: {projection_checkpoint_path}")
    # Load the trained weights for the projection layer ONLY
    proj_checkpoint = torch.load(projection_checkpoint_path, map_location="cpu") # Load to CPU first

    # Check if checkpoint is the state_dict itself or contains 'model_state_dict' key
    if "model_state_dict" in proj_checkpoint:
        state_dict_to_load = proj_checkpoint["model_state_dict"]
    else:
        state_dict_to_load = proj_checkpoint

    # Load the state dict into the projection MLP part of the model
    projection_model.projection.load_state_dict(state_dict_to_load)

    # Move the projection model (which is small and not quantized) to the GPU
    projection_model.to(device, dtype=compute_dtype).eval()
    print("SigLIP projection model loaded.")

    # Return the combined model (base + adapter) and the separate projection model
    return model_with_adapter, projection_model, phi_tokenizer, device, compute_dtype

def process_image(image_path, transform=None):
    """Loads and transforms an image."""
    image = Image.open(image_path).convert("RGB")
    if transform is None:
        # Use appropriate normalization (e.g., ImageNet or CIFAR-10 stats used during training)
        # Check dataloader.py or train_projection.py for the exact transform used.
        # Using CIFAR stats as example from dataloader.py
        transform = transforms.Compose([
            transforms.Resize((384, 384)), # Assuming 384x384 input for SigLIP
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

import torch
import torch.nn.functional as F

# Make sure patch_dynamic_cache() is defined and called somewhere before this function

def generate_text_from_image(
    model_with_adapter,
    projection_model,
    image_tensor,
    phi_tokenizer,
    device,
    compute_dtype,
    max_new_tokens=200,
    prompt="Describe the image." # Default prompt
    ):
    """ Generates text using input_ids with temporary embedding layer modification AND NO CACHE. """
    patch_dynamic_cache() # Ensure patch is applied

    # --- Get Image Embedding ---
    print("Generating image embedding...")
    with torch.no_grad():
        image_embeddings_proj = projection_model(image_tensor.to(device, dtype=compute_dtype))
        image_embedding_to_inject = image_embeddings_proj.squeeze()
        print(f"Projected image embedding shape for injection: {image_embedding_to_inject.shape}")

    # --- Prepare Placeholder and Chat Template ---
    placeholder_token = "<image>"
    if placeholder_token not in phi_tokenizer.get_vocab():
        print(f"Adding special token '{placeholder_token}' to tokenizer.")
        phi_tokenizer.add_tokens([placeholder_token], special_tokens=True)
        model_with_adapter.resize_token_embeddings(len(phi_tokenizer))
        print(f"Resized model embeddings to: {len(phi_tokenizer)}")
    placeholder_token_id = phi_tokenizer.convert_tokens_to_ids(placeholder_token)
    print(f"Placeholder token '{placeholder_token}' ID: {placeholder_token_id}")

    messages = [
        {"role": "user", "content": f"{placeholder_token}\n{prompt}"}
    ]
    try:
        formatted_prompt_ids = phi_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)
        print(f"Formatted prompt using chat template + placeholder. Shape: {formatted_prompt_ids.shape}")
        print("Formatted Prompt (decoded):", phi_tokenizer.decode(formatted_prompt_ids[0]))
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return "Error during template formatting."

    prompt_input_ids = formatted_prompt_ids
    prompt_attention_mask = torch.ones_like(prompt_input_ids)
    print(f"Attention mask shape: {prompt_attention_mask.shape}")

    # --- Modify Embedding Layer ---
    embedding_layer = model_with_adapter.get_input_embeddings()
    original_embedding = None

    try:
        original_embedding = embedding_layer.weight.data[placeholder_token_id].clone()
        embedding_layer.weight.data[placeholder_token_id] = image_embedding_to_inject.to(
            embedding_layer.weight.device, dtype=embedding_layer.weight.dtype
        )
        print(f"Temporarily injected embedding for token {placeholder_token_id}.")

        # --- Text Generation (using input_ids, NO CACHE) ---
        print("Generating text using input_ids with modified embedding layer (NO CACHE)...")
        with torch.no_grad():
            outputs = model_with_adapter.generate(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=phi_tokenizer.pad_token_id,
                eos_token_id=phi_tokenizer.eos_token_id,
                use_cache=False # <<< DISABLE CACHE
            )

    finally:
        # --- Restore Original Embedding ---
        if original_embedding is not None:
            embedding_layer.weight.data[placeholder_token_id] = original_embedding
            print("Restored original embedding for placeholder token.")

    # Decode generated text
    prompt_length = prompt_input_ids.shape[1]
    generated_ids = outputs[0][prompt_length:]
    generated_text = phi_tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Debug: Print full output
    print("Full generated output IDs:", outputs[0])
    print("Decoded generated text (only):", generated_text)

    if not generated_text and len(generated_ids) <= 1:
        print("--- WARNING: Multimodal test resulted in empty or EOS-only output. ---")

    return generated_text

def main(args):
    # Use configs imported from training scripts if available
    # Ensure these imported values match the models being loaded
    compute_dtype = QLORA_CONFIG.get("COMPUTE_DTYPE", torch.bfloat16)
    quant_type = QLORA_CONFIG.get("QUANT_TYPE", "nf4")
    projection_dim = PROJECTION_CONFIG.get("PROJECTION_DIM", 4096) # Example intermediate dim
    projection_layers = PROJECTION_CONFIG.get("PROJECTION_LAYERS", 2)

    # Setup quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True
    )

    # Load models
    # Note: If ValueError occurs here, it's likely a library version issue.
    try:
        model_with_adapter, projection_model, phi_tokenizer, device, compute_dtype = load_models_for_inference(
            phi3_model_name=args.phi3_model_name, # Use phi3 arg
            siglip_model_name=args.siglip_model_name,
            projection_checkpoint_path=args.projection_model_path,
            adapter_checkpoint_path=args.adapter_checkpoint_path,
            projection_dim=projection_dim,
            projection_layers=projection_layers,
            quant_config=quantization_config
        )
    except ValueError as e:
        print(f"\n--- ERROR DURING MODEL LOADING ---")
        print(e)
        print("This error often indicates incompatible library versions (transformers, accelerate, bitsandbytes).")
        print("Try updating libraries (`!pip install --upgrade ...`) and restarting the Colab runtime.")
        print("---------------------------------\n")
        return # Exit if model loading fails

    # Process image
    image_tensor = process_image(args.image_path)

    # Generate text
    generated_text = generate_text_from_image(
        model_with_adapter,
        projection_model,
        image_tensor,
        phi_tokenizer,
        device,
        compute_dtype,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt
    )

    # Print result
    print("\n" + "="*50)
    print(f"Generated text for image: {args.image_path}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    print(generated_text)
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with SigLIP to Phi-3 projection and QLoRA adapter")

    # Model Arguments
    parser.add_argument("--siglip_model_name", type=str, default="google/siglip-so400m-patch14-384", help="Name/path of the SigLIP model")
    parser.add_argument("--phi3_model_name", type=str, default="microsoft/phi-3-mini-4k-instruct", help="Name/path of the Phi-3 model")
    parser.add_argument("--projection_model_path", type=str, default="./siglip_phi3_projection/best_model.pt", help="Path to the trained projection model checkpoint (.pt file trained for Phi-3)")
    parser.add_argument("--adapter_checkpoint_path", type=str, default="./phi3_qlora_adapter/final_model/", help="Path to the trained QLoRA adapter directory (trained for Phi-3)")

    # Input Arguments
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, default="Describe the image in detail.", help="Text prompt to guide the generation")

    # Generation Arguments
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of new tokens to generate")

    args = parser.parse_args()
    main(args)