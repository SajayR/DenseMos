import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
import wandb  # for tracking experiments
import time
from tqdm import tqdm
from shampoo import SOAP
from dataset import AudioVisualDataset, VideoBatchSampler, collate_fn
from model import AudioVisualModel
from viz import process_video
import random
def select_random_viz_videos(data_root: str, num_samples: int = 5) -> list:
    """Select random videos from dataset for visualization"""
    video_files = list(Path(data_root).glob("*.mp4"))
    return random.sample(video_files, min(num_samples, len(video_files)))

def train(
        data_root: str = "vggsound_split_1seconds",
        batch_size: int = 20,
        num_epochs: int = 100,
        learning_rate: float = 3e-4,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        viz_interval: int = 100,
        checkpoint_interval: int = 1000,
        num_viz_samples: int = 5  # New parameter
        ):
    # Select random videos for visualization
    viz_videos = select_random_viz_videos(data_root, num_viz_samples)
    print(f"Selected videos for visualization:")
    for vid in viz_videos:
        print(f"  - {vid}")

    # Rest of the training code remains the same...
    # Initialize wandb
    wandb.init(
        project="audio-visual-alignment",
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "architecture": "cosmos_denseav"
        }
    )

    # Setup dataset and dataloader
    dataset = AudioVisualDataset(data_root)
    sampler = VideoBatchSampler(dataset.vid_nums, batch_size)
    print("Initializing dataloader")
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        #persistent_workers=True
    )
    print("Dataloader initialized")
    # Initialize model and optimizer
    print("Initializing model")
    model = AudioVisualModel().to(device)
    print("Model initialized")
    #optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = SOAP(model.parameters(), lr=learning_rate)
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    print("Starting training loop")
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader)
        for batch in pbar:
            # Move batch to device
            video_tokens = batch['video_tokens'].to(device)
            audio = batch['audio'].to(device)
            
            # Forward pass and loss computation
            loss = model(video_tokens, audio)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log loss
            epoch_losses.append(loss.item())
            wandb.log({
                "train_loss": loss.item(),
                "temperature": model.temperature.item(),
                "global_step": global_step
            })
            
            pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Visualization checkpoint
            if global_step % viz_interval == 0:
                model.eval()
                for vid_path in viz_videos:
                    viz_path = f"viz/step_{global_step}_{Path(vid_path).stem}.mp4"
                    Path("viz").mkdir(exist_ok=True)
                    process_video(model, vid_path, viz_path, device)
                    # Log video to wandb
                    wandb.log({
                        f"visualization_{Path(vid_path).stem}": wandb.Video(viz_path)
                    })
                model.train()
            
            # Save checkpoint
            if global_step % checkpoint_interval == 0 and global_step > 0:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }
                
                torch.save(
                    checkpoint,
                    checkpoint_dir / f"step_{global_step}.pt"
                )
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    torch.save(
                        checkpoint,
                        checkpoint_dir / "best_model.pt"
                    )
            
            global_step += 1
        
        # End of epoch logging
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        wandb.log({
            "epoch": epoch,
            "epoch_loss": avg_epoch_loss
        })
        
        print(f"Epoch {epoch} complete. Average loss: {avg_epoch_loss:.4f}")

if __name__ == "__main__":
    # Set up any command line arguments here if needed
    train()