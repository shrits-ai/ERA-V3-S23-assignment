import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
import wandb
import numpy as np
from dataloader import create_dataloaders

class SigLIPProjectionModel(nn.Module):
    """Model that projects SigLIP embeddings to Phi-3 embedding space."""
    
    def __init__(self, siglip_model_name, phi_model_name, projection_dim=None):
        """
        Initialize the projection model.
        
        Args:
            siglip_model_name (str): Name of the SigLIP model
            phi_model_name (str): Name of the Phi-3 model
            projection_dim (int, optional): Dimension of the projection layer
        """
        super().__init__()
        
        # Load SigLIP model
        self.siglip = AutoModel.from_pretrained(siglip_model_name)
        
        # Load Phi-3 model
        self.phi = AutoModelForCausalLM.from_pretrained(
            phi_model_name, 
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Freeze both models
        for param in self.siglip.parameters():
            param.requires_grad = False
        
        for param in self.phi.parameters():
            param.requires_grad = False
        
        # Get embedding dimensions
        siglip_dim = self.siglip.config.hidden_size
        phi_dim = self.phi.config.hidden_size
        
        # Create projection layer
        if projection_dim is None:
            projection_dim = phi_dim
        
        self.projection = nn.Sequential(
            nn.Linear(siglip_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, phi_dim)
        )
        
        # Initialize weights
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, images):
        """
        Forward pass through the model.
        
        Args:
            images (torch.Tensor): Batch of images
            
        Returns:
            torch.Tensor: Projected embeddings
        """
        # Get SigLIP embeddings
        with torch.no_grad():
            siglip_outputs = self.siglip(pixel_values=images)
            siglip_embeddings = siglip_outputs.last_hidden_state[:, 0]  # Use CLS token
        
        # Project to Phi-3 embedding space
        projected_embeddings = self.projection(siglip_embeddings)
        
        return projected_embeddings

def compute_loss(projected_embeddings, target_embeddings, loss_fn):
    """
    Compute the loss between projected embeddings and target embeddings.
    
    Args:
        projected_embeddings (torch.Tensor): Projected SigLIP embeddings
        target_embeddings (torch.Tensor): Target Phi-3 embeddings
        loss_fn: Loss function
        
    Returns:
        torch.Tensor: Loss value
    """
    return loss_fn(projected_embeddings, target_embeddings)

def extract_phi_embeddings(model, input_ids, attention_mask):
    """
    Extract embeddings from Phi-3 model.
    
    Args:
        model: Phi-3 model
        input_ids (torch.Tensor): Input token IDs
        attention_mask (torch.Tensor): Attention mask
        
    Returns:
        torch.Tensor: Phi-3 embeddings
    """
    with torch.no_grad():
        # Get the embedding layer
        embeddings = model.get_input_embeddings()(input_ids)
        
        # Average the embeddings across the sequence dimension (weighted by attention mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled_embeddings = (embeddings * mask).sum(dim=1) / mask.sum(dim=1)
        
    return pooled_embeddings

def train_epoch(model, train_loader, optimizer, scaler, loss_fn, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        # Move data to device
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            # Get projected embeddings
            projected_embeddings = model(images)
            
            # Get target Phi-3 embeddings
            target_embeddings = extract_phi_embeddings(
                model.phi, 
                input_ids, 
                attention_mask
            )
            
            # Compute loss
            loss = compute_loss(projected_embeddings, target_embeddings, loss_fn)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update progress
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(train_loader)

def validate(model, val_loader, loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Get projected embeddings
            projected_embeddings = model(images)
            
            # Get target Phi-3 embeddings
            target_embeddings = extract_phi_embeddings(
                model.phi, 
                input_ids, 
                attention_mask
            )
            
            # Compute loss
            loss = compute_loss(projected_embeddings, target_embeddings, loss_fn)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main(args):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project="siglip-phi3-projection",
            config=vars(args)
        )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        phi_processor_name=args.phi_model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    model = SigLIPProjectionModel(
        siglip_model_name=args.siglip_model_name,
        phi_model_name=args.phi_model_name,
        projection_dim=args.projection_dim
    ).to(device)
    
    # Set up optimizer
    optimizer = optim.AdamW(
        model.projection.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Set up learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate / 100
    )
    
    # Set up loss function
    loss_fn = nn.MSELoss()
    
    # Set up gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, loss_fn, device)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.projection.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, os.path.join(args.output_dir, "best_model.pt"))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.projection.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Save final model
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.projection.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }, os.path.join(args.output_dir, "final_model.pt"))
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SigLIP to Phi-3 projection layer")
    
    # Model arguments
    parser.add_argument("--siglip_model_name", type=str, default="google/siglip-so400m-patch14-384",
                        help="Name of the SigLIP model")
    parser.add_argument("--phi_model_name", type=str, default="microsoft/phi-3-mini-4k-instruct",
                        help="Name of the Phi-3 model")
    parser.add_argument("--projection_dim", type=int, default=2048,
                        help="Dimension of the projection layer")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="cifar10_vlm_dataset",
                        help="Directory containing the dataset")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--output_dir", type=str, default="siglip_phi3_projection",
                        help="Directory to save models")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use Weights & Biases for logging")
    
    args = parser.parse_args()
    main(args) 