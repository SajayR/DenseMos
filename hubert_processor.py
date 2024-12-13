import torch
import torchaudio
import av
from transformers import AutoProcessor, HubertModel
from pathlib import Path
import numpy as np

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

def get_hubert_features(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Process audio through HuBERT model and return features."""
    print(f"Processing audio tensor of shape: {audio_tensor.shape}")
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    
    # Ensure input matches HuBERT's requirements
    input_values = processor(
        audio_tensor, 
        return_tensors="pt",
        sampling_rate=16000,
        padding=True,
        return_attention_mask=True
    ).input_values

    # Move to same device as model if using GPU
    device = next(model.parameters()).device
    input_values = input_values.to(device)
    
    with torch.no_grad():
        outputs = model(input_values)
        # Shape will be [batch_size, sequence_length, hidden_size]
        features = outputs.last_hidden_state

    return features

def process_video_hubert(video_path: Path) -> torch.Tensor:
    """Extract audio from video and process through HuBERT."""
    audio = extract_audio_from_video(video_path)
    features = get_hubert_features(audio)
    return features


if __name__ == "__main__":
    # Example usage
    video_path = Path("/home/cisco/heyo/densefuck/sound_of_pixels/densetok/densefuckfuckfuck/vggsound_split_1seconds/2_0.mp4")
    features = process_video_hubert(video_path)
    print(f"Extracted features shape: {features.shape}")
