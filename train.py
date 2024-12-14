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

def find_latest_checkpoint(checkpoint_dir: str) -> tuple[Path, int]:
    """Find the latest checkpoint in the directory based on step number."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("step_*.pt"))
    
    if not checkpoints:
        return None, 0
        
    # Extract step numbers and find the latest
    steps = [int(cp.stem.split('_')[1]) for cp in checkpoints]
    latest_idx = max(range(len(steps)), key=steps.__getitem__)
    
    return checkpoints[latest_idx], steps[latest_idx]

def train(
        data_root: str = "vggsound_split_1seconds",
        batch_size: int = 20,
        num_epochs: int = 200,
        learning_rate: float = 3e-4,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        viz_interval: int = 100,
        checkpoint_interval: int = 1000,
        num_viz_samples: int = 5,
        resume_training: bool = True,  # New parameter to control checkpoint loading
        do_wandb: bool = False
        ):
    
    # Find latest checkpoint if we want to resume
    latest_checkpoint, start_step = None, 0
    if resume_training:
        latest_checkpoint, start_step = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print(f"Found checkpoint at step {start_step}: {latest_checkpoint}")
        else:
            print("No existing checkpoints found. Starting from scratch.")
    
    # Select random videos for visualization
    viz_videos = select_random_viz_videos(data_root, num_viz_samples)
    print(f"Selected videos for visualization:")
    for vid in viz_videos:
        print(f"  - {vid}")

    # Initialize wandb with resumed step
    if do_wandb:
        wandb.init(
            project="audio-visual-alignment",
            config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "architecture": "cosmos_denseav",
                "resumed_from_step": start_step
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
    )
    print("Dataloader initialized")
    
    # Initialize model and optimizer
    print("Initializing model")
    model = AudioVisualModel().to(device)
    optimizer = SOAP(model.parameters(), lr=learning_rate)
    
    # Load checkpoint if it exists
    if latest_checkpoint is not None:
        print("Loading checkpoint weights...")
        checkpoint = torch.load(latest_checkpoint)
        # Load only model weights, ignore optimizer state
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from step {start_step}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    print("Starting training loop")
    global_step = start_step  # Start from the loaded checkpoint step
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
            if do_wandb:
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
                    if do_wandb:
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
        if do_wandb:
            wandb.log({
                "epoch": epoch,
                "epoch_loss": avg_epoch_loss
            })
        
        print(f"Epoch {epoch} complete. Average loss: {avg_epoch_loss:.4f}")

if __name__ == "__main__":
    train(resume_training=True)  # Enable checkpoint loading by default