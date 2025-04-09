import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import wandb
import numpy as np
from dataloader import create_dataloaders
from train_projection import SigLIPProjectionModel

# Configuration
CONFIG = {
    # Model parameters
    "SIGLIP_MODEL": "google/siglip-so400m-patch14-384",
    "PHI_MODEL": "microsoft/phi-3-mini-4k-instruct",
    "PROJECTION_MODEL_PATH": "siglip_phi3_projection/best_model.pt",
    
    # QLoRA parameters
    "LORA_R": 16,
    "LORA_ALPHA": 32,
    "LORA_DROPOUT": 0.05,
    "LORA_TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Quantization parameters
    "QUANTIZATION_BITS": 4,  # 4-bit quantization
    "QUANT_TYPE": "nf4",     # NF4 data type
    
    # Training parameters
    "BATCH_SIZE": 4,
    "LEARNING_RATE": 2e-4,
    "WEIGHT_DECAY": 0.01,
    "EPOCHS": 3,
    "SAVE_EVERY": 1,
    "GRADIENT_ACCUMULATION_STEPS": 4,
    
    # Optimizer parameters
    "OPTIMIZER": "adamw",
    "SCHEDULER": "cosine",
    "LR_MIN_FACTOR": 0.1,   # Minimum LR will be LR * this factor
    "WARMUP_RATIO": 0.03,   # Percentage of steps for warmup
    
    # Mixed precision
    "USE_MIXED_PRECISION": True,
    
    # Data parameters
    "DATA_DIR": "cifar10_vlm_dataset",
    "NUM_WORKERS": 4,
    
    # Output parameters
    "OUTPUT_DIR": "phi3_qlora_adapter",
    "USE_WANDB": False,
}

class MultimodalPhiModel(nn.Module):
    """Model that combines SigLIP projection with Phi-3 for multimodal capabilities."""
    
    def __init__(self, siglip_model_name, phi_model_name, projection_model_path, quantization_config=None):
        """
        Initialize the multimodal model.
        
        Args:
            siglip_model_name (str): Name of the SigLIP model
            phi_model_name (str): Name of the Phi-3 model
            projection_model_path (str): Path to the trained projection model
            quantization_config: Configuration for quantization
        """
        super().__init__()
        
        # Load SigLIP projection model
        self.projection_model = SigLIPProjectionModel(
            siglip_model_name=siglip_model_name,
            phi_model_name=phi_model_name
        )
        
        # Load trained projection weights
        if projection_model_path and os.path.exists(projection_model_path):
            print(f"Loading projection model from {projection_model_path}")
            checkpoint = torch.load(projection_model_path, map_location="cpu")
            self.projection_model.projection.load_state_dict(checkpoint["model_state_dict"])
        else:
            print(f"Warning: Projection model path {projection_model_path} not found. Using untrained projection.")
        
        # Freeze the projection model
        for param in self.projection_model.parameters():
            param.requires_grad = False
        
        # Load Phi-3 model with quantization for QLoRA
        if quantization_config:
            self.phi = AutoModelForCausalLM.from_pretrained(
                phi_model_name,
                quantization_config=quantization_config,
                device_map="auto"
            )
            # Prepare model for k-bit training
            self.phi = prepare_model_for_kbit_training(self.phi)
        else:
            self.phi = AutoModelForCausalLM.from_pretrained(
                phi_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        
        # Load processor for tokenization
        self.processor = AutoProcessor.from_pretrained(phi_model_name)
    
    def forward(self, images, input_ids, attention_mask, labels=None):
        """
        Forward pass through the model.
        
        Args:
            images (torch.Tensor): Batch of images
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask
            labels (torch.Tensor, optional): Labels for computing loss
            
        Returns:
            torch.Tensor: Loss or logits
        """
        # Get projected image embeddings
        with torch.no_grad():
            image_embeddings = self.projection_model(images)
        
        # Get the embedding layer from Phi model
        input_embeddings = self.phi.get_input_embeddings()(input_ids)
        
        # Replace the first token embedding with the image embedding
        # We'll use the first token position to inject the image information
        batch_size = input_embeddings.shape[0]
        for i in range(batch_size):
            input_embeddings[i, 0, :] = image_embeddings[i]
        
        # Forward pass through Phi model
        outputs = self.phi(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs

def train_epoch(model, train_loader, optimizer, scaler, device, gradient_accumulation_steps=1):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    optimizer.zero_grad()
    
    for step, batch in enumerate(progress_bar):
        # Move data to device
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Create labels for causal language modeling (shift input_ids right)
        labels = input_ids.clone()
        # Shift right to create targets and mask the image token position (first token)
        labels[:, 0] = -100  # Don't compute loss for the image token position
        
        # Forward pass with mixed precision
        with autocast(enabled=CONFIG["USE_MIXED_PRECISION"]):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights if we've accumulated enough gradients
        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.phi.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Update progress
        total_loss += loss.item() * gradient_accumulation_steps
        progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Create labels for causal language modeling
            labels = input_ids.clone()
            labels[:, 0] = -100  # Don't compute loss for the image token position
            
            # Forward pass
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main(args):
    # Print configuration
    print("Running with configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project="phi3-qlora-multimodal",
            config=vars(args)
        )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        phi_processor_name=args.phi_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print(f"Created dataloaders: {len(train_loader)} training batches, {len(val_loader)} validation batches")
    
    # Set up quantization config for QLoRA
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=args.quant_type,
        bnb_4bit_use_double_quant=True
    )
    
    # Create model
    model = MultimodalPhiModel(
        siglip_model_name=args.siglip_model_name,
        phi_model_name=args.phi_model_name,
        projection_model_path=args.projection_model_path,
        quantization_config=quantization_config
    )
    
    # Apply LoRA to the model
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model.phi = get_peft_model(model.phi, lora_config)
    model.phi.print_trainable_parameters()
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.phi.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    if args.scheduler.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=args.learning_rate * args.lr_min_factor
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=args.learning_rate * args.lr_min_factor
        )
    
    # Set up gradient scaler for mixed precision
    scaler = GradScaler(enabled=args.use_mixed_precision)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            scaler, 
            device, 
            gradient_accumulation_steps=args.gradient_accumulation_steps
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.phi.save_pretrained(os.path.join(args.output_dir, "best_model"))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            model.phi.save_pretrained(os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}"))
    
    # Save final model
    model.phi.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Phi-3 with SigLIP projection")
    
    # Model arguments
    parser.add_argument("--siglip_model_name", type=str, default=CONFIG["SIGLIP_MODEL"],
                        help="Name of the SigLIP model")
    parser.add_argument("--phi_model_name", type=str, default=CONFIG["PHI_MODEL"],
                        help="Name of the Phi-3 model")
    parser.add_argument("--projection_model_path", type=str, default=CONFIG["PROJECTION_MODEL_PATH"],
                        help="Path to the trained projection model")
    
    # QLoRA arguments
    parser.add_argument("--lora_r", type=int, default=CONFIG["LORA_R"],
                        help="Rank of the LoRA update matrices")
    parser.add_argument("--lora_alpha", type=int, default=CONFIG["LORA_ALPHA"],
                        help="Scaling factor for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=CONFIG["LORA_DROPOUT"],
                        help="Dropout probability for LoRA layers")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=CONFIG["LORA_TARGET_MODULES"],
                        help="List of module names to apply LoRA to")
    parser.add_argument("--quant_type", type=str, default=CONFIG["QUANT_TYPE"],
                        help="Quantization type (nf4 or fp4)")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default=CONFIG["DATA_DIR"],
                        help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"],
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=CONFIG["NUM_WORKERS"],
                        help="Number of workers for data loading")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=CONFIG["EPOCHS"],
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=CONFIG["LEARNING_RATE"],
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=CONFIG["WEIGHT_DECAY"],
                        help="Weight decay")
    parser.add_argument("--save_every", type=int, default=CONFIG["SAVE_EVERY"],
                        help="Save checkpoint every N epochs")
    parser.add_argument("--optimizer", type=str, default=CONFIG["OPTIMIZER"],
                        help="Optimizer to use (adamw, adam)")
    parser.add_argument("--scheduler", type=str, default=CONFIG["SCHEDULER"],
                        help="Learning rate scheduler (cosine, linear)")
    parser.add_argument("--lr_min_factor", type=float, default=CONFIG["LR_MIN_FACTOR"],
                        help="Minimum learning rate factor")
    parser.add_argument("--warmup_ratio", type=float, default=CONFIG["WARMUP_RATIO"],
                        help="Percentage of steps for warmup")
    parser.add_argument("--use_mixed_precision", action="store_true", default=CONFIG["USE_MIXED_PRECISION"],
                        help="Whether to use mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=CONFIG["GRADIENT_ACCUMULATION_STEPS"],
                        help="Number of steps to accumulate gradients")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default=CONFIG["OUTPUT_DIR"],
                        help="Directory to save models")
    parser.add_argument("--use_wandb", action="store_true", default=CONFIG["USE_WANDB"],
                        help="Whether to use Weights & Biases for logging")
    
    args = parser.parse_args()
    main(args) 