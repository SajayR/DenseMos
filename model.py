import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import HubertModel
from typing import Tuple, List, Optional

class DenseAVTokenized(nn.Module):
    def __init__(
        self,
        token_embed_dim: int = 1024,
        num_visual_tokens: int = 1024,  # 32*32
        vocab_size: int = 65536,  # Cosmos vocab size
        hubert_model: Optional[HubertModel] = None
    ):
        super().__init__()
        
        # Token embedding for Cosmos tokens
        self.token_embedding = nn.Embedding(vocab_size, token_embed_dim)
        
        # HuBERT model
        self.hubert = hubert_model or HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        
        # Projections if needed
        self.token_projection = nn.Linear(token_embed_dim, token_embed_dim)
        self.audio_projection = nn.Linear(1024, token_embed_dim)  # HuBERT dim -> token_embed_dim
        
    def split_features(self, visual_features: torch.Tensor, audio_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split features into 3 second-long segments
        
        Args:
            visual_features: (B, 9, 32, 32)
            audio_features: (B, 150, 1024)
            
        Returns:
            Tuple of:
                visual_splits: (B, 3, 3, 32, 32)
                audio_splits: (B, 3, 50, 1024)
        """
        B = visual_features.shape[0]
        
        # Split visual features
        visual_splits = visual_features.view(B, 3, 3, 32, 32)
        
        # Split audio features - 150 -> 3 segments of 50
        audio_splits = audio_features.view(B, 3, 50, -1)
        
        return visual_splits, audio_splits
        
    def select_random_frames(self, visual_splits: torch.Tensor) -> torch.Tensor:
        """
        Randomly select one frame from each second-segment of visual features
        
        Args:
            visual_splits: (B, 3, 3, 32, 32) # B, num_segments, frames_per_segment, H, W
            
        Returns:
            selected_frames: (B, 3, 1, 32, 32) # One frame per segment
        """
        B = visual_splits.shape[0]
        device = visual_splits.device
        
        # Generate random indices for each batch and segment
        random_indices = torch.randint(0, 3, (B, 3, 1), device=device)
        
        # Create proper indexing tensors
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, 3, 1)
        segment_idx = torch.arange(3, device=device).view(1, 3, 1).expand(B, 3, 1)
        
        # Select random frames
        selected_frames = visual_splits[batch_idx, segment_idx, random_indices]
        
        return selected_frames  # (B, 3, 1, 32, 32)

    def embed_visual_tokens(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        """
        Embed and project visual tokens
        
        Args:
            visual_tokens: (B, 3, 1, 32, 32) 
            
        Returns:
            token_embeddings: (B, 3, 1024, 1024) # B, segments, num_tokens (flattened 32*32), embedding_dim
        """
        B = visual_tokens.shape[0]
        
        # Flatten spatial dimensions
        flat_tokens = visual_tokens.view(B, 3, -1)  # (B, 3, 1024)
        
        # Embed tokens
        token_embeddings = self.token_embedding(flat_tokens)  # (B, 3, 1024, embed_dim)
        
        # Project if needed
        token_embeddings = self.token_projection(token_embeddings)  # (B, 3, 1024, 1024)
        
        return token_embeddings
    
    def calculate_similarity(
        self, 
        visual_embeddings: torch.Tensor,  # (B, 3, 1024, 1024)
        audio_features: torch.Tensor,     # (B, 3, 50, 1024)
    ) -> torch.Tensor:
        """
        Calculate similarity between visual and audio features for each segment
        Returns similarity scores for each batch item
        """
        B = visual_embeddings.shape[0]
        
        # Compute dot product between visual and audio features
        # For each segment in batch:
        # visual_embeddings: (1024, 1024) @ audio_features: (50, 1024).T
        # = (1024, 50)
        similarities = torch.matmul(
            visual_embeddings,  # (B, 3, 1024, 1024)
            audio_features.transpose(-1, -2)  # (B, 3, 1024, 50)
        )  # Result: (B, 3, 1024, 50)
        
        # Max pool over visual tokens
        token_maxpool = torch.max(similarities, dim=2)[0]  # (B, 3, 50)
        
        # Average over audio temporal dimension
        segment_scores = torch.mean(token_maxpool, dim=2)  # (B, 3)
        
        return segment_scores  # (B, 3)

    def forward(self, visual_tokens: torch.Tensor, audio_raw: torch.Tensor, training: bool = True):
        """
        Full forward pass including similarity calculation
        """
        # Previous steps remain the same...
        # Pass audio through HuBERT
        audio_features = self.hubert(audio_raw).last_hidden_state  # (B, 150, 1024)
        audio_features = self.audio_projection(audio_features)  # (B, 150, 1024)
        
        # Split features
        visual_splits, audio_splits = self.split_features(visual_tokens, audio_features)
        
        if training:
            # Get random frames and embeddings
            selected_frames = self.select_random_frames(visual_splits)
            visual_embeddings = self.embed_visual_tokens(selected_frames)
            
            # Calculate similarity scores
            similarity_scores = self.calculate_similarity(visual_embeddings, audio_splits)
            
            # similarity_scores will be (B, 3) where each value represents
            # how well the audio and visual features match for each segment
            return similarity_scores
        else:
            # For inference, we might want to return the full similarity maps
            # We'll implement this separately
            pass



if __name__ == "__main__":
    # Create dummy model
    model = DenseAVTokenized()
    model.eval()  # Prevent HuBERT from complaining about batch norm
    
    # Create dummy batch
    B = 2  # batch size
    
    # Create dummy visual tokens (B, 9, 32, 32)
    visual_tokens = torch.randint(0, 65536, (B, 9, 32, 32))
    print(f"Visual tokens shape: {visual_tokens.shape}")
    # Create dummy audio (B, audio_len)
    audio_raw = torch.randn(B, 48281)  # 3 seconds at 16kHz
    print(f"Audio raw shape: {audio_raw.shape}")
    
    
    print("\n=== Testing Feature Splitting ===")
    # First test split_features
    with torch.no_grad():
        audio_features = model.hubert(audio_raw).last_hidden_state
        visual_splits, audio_splits = model.split_features(visual_tokens, audio_features)
    
    print(f"Visual splits shape: {visual_splits.shape}")  # Should be (B, 3, 3, 32, 32)
    print(f"Audio splits shape: {audio_splits.shape}")    # Should be (B, 3, 50, 1024)
    
    print("\n=== Testing Random Frame Selection ===")
    # Test random frame selection
    selected_frames = model.select_random_frames(visual_splits)
    print(f"Selected frames shape: {selected_frames.shape}")  # Should be (B, 3, 1, 32, 32)
    
    print("\n=== Testing Visual Token Embedding ===")
    # Test token embedding
    visual_embeddings = model.embed_visual_tokens(selected_frames)
    print(f"Visual embeddings shape: {visual_embeddings.shape}")  # Should be (B, 3, 1024, 1024)
    
    print("\n=== Testing Full Forward Pass ===")
    # Test full forward pass
    with torch.no_grad():
        outputs = model(visual_tokens, audio_raw)
        print("Forward pass successful!")
        
    # Basic sanity checks
    print("\n=== Running Sanity Checks ===")
    assert visual_splits.shape == (B, 3, 3, 32, 32), "Wrong visual split shape!" #3 seconds, 3 frames for each seconds
    assert audio_splits.shape[:-1] == (B, 3, 50), "Wrong audio split shape!" # 3 seconds, 50 features
    assert selected_frames.shape == (B, 3, 1, 32, 32), "Wrong selected frames shape!" # 3 seconds, 1 frame for each seconds
    assert visual_embeddings.shape == (B, 3, 1024, 1024), "Wrong embedding shape!" # 3 seconds, 1024 tokens, 1024 embedding dim
    
    print("All tests passed! 🎉")