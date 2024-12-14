import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
import numpy as np
import random
import av
from typing import Dict, List
import torch.nn as nn
import torchaudio.transforms as T
from cosmos_tokenizer.video_lib import CausalVideoTokenizer
import warnings
warnings.filterwarnings("ignore")
#from hubert_processor import extract_audio_from_video
import multiprocessing
from tqdm import tqdm
import torchvision
multiprocessing.set_start_method('spawn', force=True)
import av

def extract_audio_from_video(video_path: Path) -> torch.Tensor:
    """Extract audio from video file and return as tensor."""
    container = av.open(str(video_path))
    audio = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=16000)
    
    # Read all audio frames
    samples = []
    for frame in container.decode(audio):
        frame.pts = None
        frame = resampler.resample(frame)[0]
        samples.append(frame.to_ndarray().reshape(-1))
    
    samples = torch.tensor(np.concatenate(samples))
    samples = samples.float() / 32768.0  # Convert to float and normalize
    return samples

class VideoBatchSampler(Sampler):
    def __init__(self, vid_nums: List[int], batch_size: int):
        self.vid_nums = np.array(vid_nums)
        self.batch_size = batch_size
        
        # Group indices by vid_num
        self.vid_to_indices = {}
        for i, vid in enumerate(vid_nums):
            if vid not in self.vid_to_indices:
                self.vid_to_indices[vid] = []
            self.vid_to_indices[vid].append(i)
            
    def __iter__(self):
        # Get unique vid_nums
        unique_vids = list(self.vid_to_indices.keys())
        random.shuffle(unique_vids)  # Shuffle at epoch start
        
        while len(unique_vids) >= self.batch_size:
            batch_vids = unique_vids[:self.batch_size]
            
            # For each selected video, randomly pick one of its segments
            batch = []
            for vid in batch_vids:
                idx = random.choice(self.vid_to_indices[vid])
                batch.append(idx)
            
            yield batch
            unique_vids = unique_vids[self.batch_size:]
    
    def __len__(self):
        return len(set(self.vid_nums)) // self.batch_size


class AudioVisualDataset(Dataset):
    def __init__(self, data_root: str, sample_fps: int = 12):
        self.data_root = Path(data_root)
        self.cache_dir = Path("cosmos_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.sample_fps = sample_fps
        self.video_files = sorted(list(self.data_root.glob("*.mp4")))
        
        # Create vid_num mapping (same as before)
        self.vid_to_files = {}
        for file in self.video_files:
            vid_num = int(file.stem.split('_')[0])
            if vid_num not in self.vid_to_files:
                self.vid_to_files[vid_num] = []
            self.vid_to_files[vid_num].append(file)
            
        self.vid_nums = [int(f.stem.split('_')[0]) for f in self.video_files]
        
        # Initialize cache if needed
        self._initialize_cache()

    def _initialize_cache(self, batch_size=64):
        """Pre-compute and cache Cosmos encodings if not already cached, with batched processing"""
        # First check which videos need encoding
        uncached_videos = []
        for video_path in tqdm(self.video_files, desc="Checking for uncached videos"):
            cache_path = self.cache_dir / f"{video_path.stem}_tokens.pt"
            if not cache_path.exists():
                uncached_videos.append(video_path)
        
        if not uncached_videos:
            print("All videos already cached!")
            return
            
        print(f"Need to cache {len(uncached_videos)} videos")
        
        # Initialize encoder
        cosmos_encoder = CausalVideoTokenizer(
            checkpoint_enc=f'pretrained_ckpts/Cosmos-Tokenizer-DV4x8x8/encoder.jit'
        ).to('cuda')
        
        # Process in batches
        for i in tqdm(range(0, len(uncached_videos), batch_size), desc="Caching Cosmos encodings"):
            batch_videos = uncached_videos[i:i + batch_size]
            if i == 0:
                print(batch_videos)

            # Load and preprocess batch
            batch_frames = []
            for video_path in batch_videos:
                frames = load_and_preprocess_video(video_path, self.sample_fps)
                frames = frames[:, :12, :, :]  # Limit to 12 frames as in collate_fn
                batch_frames.append(frames)
            
            # Stack into batch
            batch_frames = torch.stack(batch_frames)
            
            # Encode batch
            #with torch.no_grad():
            batch_tokens = cosmos_encoder.encode(batch_frames.to('cuda'))[0]
            batch_tokens = batch_tokens[:, 1:, :, :]  # Drop first frame
            

            if i == 0:
                print(batch_tokens.shape)
            # Save individual results
            for idx, video_path in enumerate(batch_videos):
                cache_path = self.cache_dir / f"{video_path.stem}_tokens.pt"
                torch.save(batch_tokens[idx].cpu(), cache_path)
                
            # Optional: Clear CUDA cache periodically
            if i % (batch_size * 100) == 0:
                torch.cuda.empty_cache()
        
        del cosmos_encoder  # Free up GPU memory
        torch.cuda.empty_cache()

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        # Load cached tokens
        cache_path = self.cache_dir / f"{video_path.stem}_tokens.pt"
        tokens = torch.load(cache_path)
        
        audio = extract_audio_from_video(video_path)
        
        return {
            'video_path': video_path,
            'video_tokens': tokens,  # Now pre-loaded from cache
            'audio': audio,
            'vid_num': int(video_path.stem.split('_')[0]),
            'segment_num': int(video_path.stem.split('_')[1])
        }

def load_and_preprocess_video(video_path: str, sample_fps: int) -> torch.Tensor:
    """Load video and preprocess frames for Cosmos encoder.
    Returns: tensor of shape [C, T, H, W] ready for Cosmos"""
    
    # Load video and get metadata
    video = torchvision.io.read_video(video_path)
    frames, _, video_meta = video
    original_fps = video_meta["video_fps"]
    
    # Calculate frame indices for desired FPS
    video_duration = len(frames) / original_fps
    desired_frame_count = int(video_duration * sample_fps)
    
    # Sample frames at exact timestamps
    frame_indices = np.linspace(0, len(frames)-1, desired_frame_count, dtype=int)
    sampled_frames = frames[frame_indices]
    
    # Normalize to [-1, 1]
    processed_frames = sampled_frames.float() / 255.0 * 2 - 1
    
    # Resize frames to 256x256
    resized_frames = []
    for frame in processed_frames:
        resized = torch.nn.functional.interpolate(
            frame.permute(2, 0, 1).unsqueeze(0),  # [1, C, H, W]
            size=(256, 256),
            mode='bilinear',
            align_corners=False
        )
        resized_frames.append(resized.squeeze(0))
    
    # Stack frames along time dimension
    processed_frames = torch.stack(resized_frames, dim=1)  # [C, T, H, W]
    
    return processed_frames

# Global Cosmos encoder for use in collate_fn
cosmos_encoder = CausalVideoTokenizer(
    checkpoint_enc=f'pretrained_ckpts/Cosmos-Tokenizer-DV4x8x8/encoder.jit'
).to('cuda')

import random

def collate_fn(batch):
    # Get all tokens (already processed)
    video_tokens = torch.stack([item['video_tokens'] for item in batch])
    # Take a random frame from each video's tokens
    video_tokens = video_tokens[:, random.randint(0, video_tokens.shape[1]-1), :, :]
    
    # Pad audio sequences (same as before)
    max_audio_len = max(item['audio'].shape[0] for item in batch)
    audio_padded = torch.zeros(len(batch), max_audio_len)
    for i, item in enumerate(batch):
        audio_len = item['audio'].shape[0]
        audio_padded[i, :audio_len] = item['audio']
    
    return {
        'video_tokens': video_tokens,
        'audio': audio_padded,
        'vid_nums': [item['vid_num'] for item in batch],
        'segment_nums': [item['segment_num'] for item in batch]
    }

if __name__ == "__main__":
    # Test dataset and dataloader
    dataset = AudioVisualDataset("vggsound_split_1seconds")
    sampler = VideoBatchSampler(dataset.vid_nums, batch_size=8) #batch 8 needs about 2166 MB VRAM
    dataloader = DataLoader(
        dataset, 
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0,
        #persistent_workers=True 
    )
    import time
    # Test a batch
    for i in range(9999999):
        start = time.time()
        batch = next(iter(dataloader))
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"{k} shape:", v.shape)
            else:
                print(f"{k}:", v)
     
