import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms.functional import resize
from model import AudioVisualModel
from dataset import load_and_preprocess_video, extract_audio_from_video
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import random
def create_heatmap_overlay(frame, heatmap, alpha=0.6):
    """Create a heatmap overlay on the original frame"""
    # Convert heatmap to colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap), 
        cv2.COLORMAP_JET
    )
    
    # Convert frame to BGR if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame
        
    # Create overlay
    overlay = cv2.addWeighted(
        frame_bgr, 1-alpha,
        heatmap_colored, alpha, 
        0
    )
    return overlay

def process_video(model, video_path, output_path, device='cuda'):
    """
    Process a video and generate heatmap overlay visualization
    
    Args:
        model: Trained AudioVisualModel
        video_path: Path to input video
        output_path: Path to save visualization
        device: Device to run inference on
    """
    # Load and preprocess video
    cosmos_encoder = CausalVideoTokenizer(
        checkpoint_enc=f'pretrained_ckpts/Cosmos-Tokenizer-DV4x8x8/encoder.jit'
    ).to('cuda')
    frames = load_and_preprocess_video(video_path, sample_fps=12)
    audio = extract_audio_from_video(video_path)
    
    # Move to device
    frames = frames.to(device)
    audio = audio.to(device).unsqueeze(0)  # Add batch dim
    
    # Get original video for visualization
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(output_path), 
        fourcc, 
        50.0, # 50 FPS for smooth visualization
        (width, height)
    )
    
    # Run inference
    model.eval()
    with torch.no_grad():
        # Get similarity matrices (B, Na, 32, 32)
        print("During viz")
        print("Frames shape: ", frames.shape)
        print("Audio shape: ", audio.shape)
        frames=frames.unsqueeze(0)
        video_tokens = cosmos_encoder.encode(frames.to('cuda'))[0]
        video_tokens = video_tokens[:, 1:, :, :]  # Drop first frame as before
        #taking a random frame 
        video_tokens=video_tokens[:,random.randint(0,video_tokens.shape[1]-1) , :, :]
        token_sims = model(video_tokens, audio)
        
        # Process each audio timestep (Na=50)
        for t in range(token_sims.shape[1]):
            # Get current heatmap 
            heatmap = token_sims[0, t].cpu().numpy()  # (32, 32)
            
            # Normalize to [0,1]
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            
            # Upsample to original resolution
            heatmap = cv2.resize(
                heatmap, 
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )
            
            # Read original frame
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create and write overlay
            overlay = create_heatmap_overlay(frame, heatmap)
            out.write(overlay)
    
    # Cleanup
    cap.release()
    out.release()

if __name__ == "__main__":
    # Create dummy similarity matrices
    video_path = "vggsound_split_1seconds/0_0.mp4"
    
    # Get video info
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Create random attention maps
    # Shape: (1, 50, 32, 32) - batch_size=1, 50 timesteps, 32x32 spatial
    dummy_similarities = torch.randn(1, 50, 32, 32)
    # Optional: Add some structure to test visualization better
    x = torch.linspace(-2, 2, 32)
    y = torch.linspace(-2, 2, 32)
    X, Y = torch.meshgrid(x, y)
    R = torch.sqrt(X**2 + Y**2)
    
    # Create moving Gaussian attention
    for t in range(50):
        center_x = 16 + 8 * torch.sin(torch.tensor(t/10))
        center_y = 16 + 8 * torch.cos(torch.tensor(t/10))
        center_R = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
        dummy_similarities[0, t] = torch.exp(-center_R/3)

    # Create output path
    output_path = "test_visualization.mp4"
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path, 
        fourcc, 
        50.0,  # 50 FPS 
        (width, height)
    )
    
    # Process each frame
    cap = cv2.VideoCapture(video_path)
    for t in range(50):  # 50 audio timesteps
        # Get heatmap for this timestep
        heatmap = dummy_similarities[0, t].numpy()
        
        # Normalize to [0,1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # Upsample to match video resolution
        heatmap = cv2.resize(
            heatmap, 
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Read original frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create and write overlay
        overlay = create_heatmap_overlay(frame, heatmap)
        out.write(overlay)
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"Created visualization at {output_path}")



# cp vggsound_split_1seconds/0_0.mp4 .