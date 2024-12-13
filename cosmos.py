import torch
import torchvision
import cv2
import numpy as np
from pathlib import Path
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

def encode_video(
    video_path: str, 
    sample_fps: int, 
    max_frames: int = None,
    #model_name: str = "Cosmos-Tokenizer-DV4x8x8"
    encoder=None,
):
    # Load video and get metadata
    video = torchvision.io.read_video(video_path)
    frames, _, video_meta = video
    original_fps = video_meta["video_fps"]
    
    # Calculate exact frame indices for desired FPS
    video_duration = len(frames) / original_fps  # in seconds
    desired_frame_count = int(video_duration * sample_fps)
    
    # Get first frame and then sample the rest
    first_frame = frames[0:1]  # Keep dimension
    
    # Sample remaining frames at exact timestamps
    # Subtract 1 from desired count since we already have first frame
    remaining_frame_indices = np.linspace(0, len(frames)-1, desired_frame_count, dtype=int)
    sampled_frames = frames[remaining_frame_indices]
    
    # Concatenate first frame with sampled frames
    final_frames = torch.cat([first_frame, sampled_frames])
    
    # Apply frame limit if specified
    if max_frames is not None:
        final_frames = final_frames[:max_frames]
        
    #print(f"Video duration: {video_duration:.2f}s")
    #print(f"Original frames: {len(frames)}")
    #print(f"Final frames (including standalone): {len(final_frames)}")
    
    # Rest of your processing code...
        
    #print(f"Processing {len(sampled_frames)} frames...")
    
    processed_frames = sampled_frames.float() / 255.0 * 2 - 1  
    
    resized_frames = []
    for frame in processed_frames:
        resized = torch.nn.functional.interpolate(
            frame.permute(2, 0, 1).unsqueeze(0),
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        resized_frames.append(resized.squeeze(0))
        
    processed_frames = torch.stack(resized_frames, dim=1)
    processed_frames = processed_frames.unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder#CausalVideoTokenizer(
        #checkpoint_enc=f'/home/cisco/heyo/densefuck/sound_of_pixels/densetok/pretrained_ckpts/{model_name}/encoder.jit'
    #).to(device)

    # Process all frames at once
    processed_frames = processed_frames.to(device)
    #print(f"Input shape: {processed_frames.shape}") # torch.Size([1, 3, 36, 256, 256]) B, C, T, H, W
    indices, _ = encoder.encode(processed_frames)
    #print(indices.shape) # (1, 10, 32, 32)
    indices = indices[:, 1:, :, :] #dropping the first frame
    return indices

if __name__ == "__main__":
    video_path = "/home/cisco/heyo/densefuck/sound_of_pixels/densetok/densefuckfuckfuck/vggsound_split/0_0.mp4"
    sample_fps = 12
    max_frames =  37
    #output_path = encode_video(video_path, sample_fps, max_frames)
    import tqdm
    num_vids_in_split = len(list(Path("/home/cisco/heyo/densefuck/sound_of_pixels/densetok/densefuckfuckfuck/vggsound_split").glob("*.mp4")))
    for vid in tqdm.tqdm(Path("/home/cisco/heyo/densefuck/sound_of_pixels/densetok/densefuckfuckfuck/vggsound_split").glob("*.mp4"), total=num_vids_in_split):
        print(vid)
        indices = encode_video(vid, sample_fps, max_frames)
        print(indices.shape)
    
