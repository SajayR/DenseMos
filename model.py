import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import transformers
from transformers import HubertModel, AutoProcessor

class AudioEmbedder(nn.Module):
    def __init__(self, embedding_dim=1024):
        super().__init__()
        
        # Load pretrained HuBERT and processor
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
        
        # Project HuBERT features (1024 for large model) to desired embedding dimension
        self.projection = nn.Linear(1024, embedding_dim)  
        
    def forward(self, audio_input):
        """
        Args:
            audio_input: (B, T) raw audio waveform at 16kHz
            
        Returns:
            features: (B, Na, D) where:
                B is batch size
                Na is number of audio tokens
                D is embedding_dim
        """
        # Process audio through HuBERT processor
        #print(f"Audio input shape: {audio_input.shape}")
        inputs = self.processor(
            audio_input, 
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True
        ).input_values.squeeze(0)
        #print(f"Inputs shape: {inputs.shape}")
        
        # Move to same device as model
        inputs = inputs.to(audio_input.device)
        
        # Get HuBERT features
        with torch.no_grad():
            hubert_output = self.hubert(inputs).last_hidden_state  # (B, T/320, 1024)
        
        # Project to embedding dimension
        features = self.projection(hubert_output)  # (B, T/320, embedding_dim)
        
        return features

class VisualEmbedder(nn.Module):
    def __init__(self, num_tokens=65536, embedding_dim=1024):  # Cosmos uses 64K vocab
        super().__init__()
        
        # Token embedding
        self.token_embedding = nn.Embedding(num_tokens, embedding_dim)
        
        # 2D positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 32*32, embedding_dim))
        
    def forward(self, token_indices):
        """
        Args:
            token_indices: (B, H, W) tensor of Cosmos token indices
            
        Returns:
            features: (B, H*W, D) embedded tokens with positional encoding
        """
        B, H, W = token_indices.shape
        
        # Embed tokens
        tokens = self.token_embedding(token_indices)  # (B, H, W, D)
        tokens = tokens.view(B, H*W, -1)  # (B, H*W, D)
        
        # Add positional embeddings
        tokens = tokens + self.pos_embedding
        
        return tokens


class AudioVisualModel(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        
        self.visual_embedder = VisualEmbedder()
        self.audio_embedder = AudioEmbedder()
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def compute_similarity_matrix(self, audio_feats, visual_feats): #ye take this
        """
        Compute pairwise cosine similarities between audio and visual tokens
        
        Args:
            audio_feats: (B, Na, D)  # B=batch, Na=num_audio_tokens, D=embedding_dim
            visual_feats: (B, Nv, D) # Nv=num_visual_tokens
            
        Returns:
            similarity_matrix: (B, Na, Nv)
        """
        # Normalize embeddings
        #print("Audio feats stats before norm - min:", audio_feats.min().item(), "max:", audio_feats.max().item())
        #print("Visual feats stats before norm - min:", visual_feats.min().item(), "max:", visual_feats.max().item())
        
        # Normalize embeddings
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute similarities and check values
        similarity = torch.bmm(audio_feats, visual_feats.transpose(1, 2))
        #print("Raw similarity stats - min:", similarity.min().item(),
          #  "max:", similarity.max().item())
        
        return similarity / self.temperature
    
    def aggregate_token_similarities(self, similarity_matrix): #also take this
        """
        Aggregate token-level similarities using max-mean strategy
        
        Args:
            similarity_matrix: (B, Na, Nv)
            
        Returns:
            clip_similarity: (B)
        """
        # Max pool over visual dimension for each audio token
        max_similarities = torch.max(similarity_matrix, dim=2)[0]  # (B, Na)
        
        # Average over audio tokens
        clip_similarity = torch.mean(max_similarities, dim=1)  # (B)
        return clip_similarity
    
    def compute_all_similarities(self, audio_feats, visual_feats):
        """Compute similarities between all pairs of audio and visual features in batch"""
        B = audio_feats.shape[0]
        
        audio_feats = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)
        visual_feats = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Normalize features
        audio_feats = F.normalize(audio_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # Compute token-level similarities
        token_sims = torch.matmul(
            audio_feats, 
            visual_feats.transpose(2, 3)
        ) / self.temperature
        
        # Aggregate using max-mean strategy
        max_sims = torch.max(token_sims, dim=3)[0]  # Max over visual dimension (B, B, Na)
        clip_sims = torch.mean(max_sims, dim=2)     # Mean over audio dimension (B, B)
        
        return clip_sims, token_sims

    '''def compute_contrastive_loss(self, clip_similarities, token_sims):
        """Compute InfoNCE loss with regularization"""
        batch_size = clip_similarities.shape[0]
        labels = torch.arange(batch_size).to(clip_similarities.device)
        
        # Audio to Visual direction
        log_prob_a2v = F.log_softmax(clip_similarities, dim=1)
        losses_a2v = -log_prob_a2v[torch.arange(batch_size), labels]
        
        # Visual to Audio direction  
        log_prob_v2a = F.log_softmax(clip_similarities.t(), dim=1)
        losses_v2a = -log_prob_v2a[torch.arange(batch_size), labels]
        
        # Average both directions
        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        
        # Add regularization
        reg_loss = self.compute_regularization_losses(clip_similarities, token_sims)
        
        total_loss = contrastive_loss + reg_loss
        
        return total_loss'''
    
    def compute_contrastive_loss(self, clip_similarities, token_sims):
        """Compute InfoNCE loss with regularization"""
        batch_size = clip_similarities.shape[0]
        labels = torch.arange(batch_size).to(clip_similarities.device)
        
        # Scale the similarities by sqrt(batch_size)
        scaled_sims = clip_similarities / math.sqrt(batch_size)
        
        # Audio to Visual direction
        log_prob_a2v = F.log_softmax(scaled_sims, dim=1)
        losses_a2v = -log_prob_a2v[torch.arange(batch_size), labels]
        
        # Visual to Audio direction  
        log_prob_v2a = F.log_softmax(scaled_sims.t(), dim=1)
        losses_v2a = -log_prob_v2a[torch.arange(batch_size), labels]
        
        # Average both directions
        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        
        # Add regularization
        reg_loss = self.compute_regularization_losses(scaled_sims, token_sims)
        
        total_loss = contrastive_loss + reg_loss
        
        return total_loss
    
    def compute_regularization_losses(self, clip_sims, token_sims):
        """Compute essential regularization terms:
        1. Non-negative pressure - encourage positive evidence
        2. Temperature stability - prevent collapse"""
        
        # Non-negative pressure on token similarities
        neg_sims = torch.clamp(token_sims, max=0)  
        l_nonneg = torch.mean(neg_sims ** 2)
        
        # Temperature/Calibration stability
        l_cal = torch.clamp(torch.log(torch.tensor(1.0, device=token_sims.device)) - 
                        torch.log(self.temperature), min=0) ** 2
        
        # Combine with smaller weights since we removed the other terms
        reg_loss = 0.01 * l_nonneg + 0.1 * l_cal
                    
        return reg_loss
    
    '''def compute_regularization_losses(self, clip_sims, token_sims):
        """Compute essential regularization terms:
        1. Non-negative pressure - encourage positive evidence
        2. Temperature stability - keep temperature in good range"""
        
        # Non-negative pressure on token similarities
        neg_sims = torch.clamp(token_sims, max=0)  
        l_nonneg = torch.mean(neg_sims ** 2)
        
        # Temperature/Calibration stability
        # Penalize if temp goes below 0.07 or above 0.2
        l_cal_lower = torch.clamp(torch.log(torch.tensor(0.07, device=token_sims.device)) - 
                        torch.log(self.temperature), min=0) ** 2
        l_cal_upper = torch.clamp(torch.log(self.temperature) - 
                        torch.log(torch.tensor(0.2, device=token_sims.device)), min=0) ** 2
        l_cal = l_cal_lower + l_cal_upper
        
        # Combine with smaller weights
        reg_loss = 0.01 * l_nonneg + 0.1 * l_cal
                    
        return reg_loss'''
        
    def forward(self, frames, audio):
        """
        Forward pass computing embeddings, similarities and loss
        
        Args:
            frames: (B, 1, H, W) batch of video frames from cosmos tokenizer #should be (b, 32, 32)
            audio: (B, T) batch of audio features #should be (b, 16331)
            
        Returns:
            loss if training, clip_similarities if not
        """
        # Get embeddings
        visual_feats = self.visual_embedder(frames)
        audio_feats = self.audio_embedder(audio)
        
        if self.training:
            # Get similarities and token-level similarities
            clip_sims, token_sims = self.compute_all_similarities(audio_feats, visual_feats)
            return self.compute_contrastive_loss(clip_sims, token_sims)
        else:
            # During inference, just get clip similarities
            token_sims = self.compute_similarity_matrix(audio_feats, visual_feats) #(B, Na, Nv)
            #should be reshaped into (B, Na, 32, 32) to bring back to original shape
            B, Na, Nv = token_sims.shape
            token_sims = token_sims.view(B, Na, 32, 32)
            return token_sims
        

def test_contrastive_loss():
    # Create a sample similarity matrix where diagonal (matching pairs)
    # should have higher values (closer to 1) than non-matching pairs
    #clip_sims = torch.tensor([
    #    [0.8, 0.2, 0.1],  # audio0 matches best with visual0
    #    [0.3, 0.9, 0.2],  # audio1 matches best with visual1
    #    [0.2, 0.1, 0.7]   # audio2 matches best with visual2
    #])
    clip_sims = torch.tensor([
        [0, 0, 1.0],  # audio0 matches best with visual0
        [1.0, 0, 0],  # audio1 matches best with visual1
        [1.0, 0, 0]   # audio2 matches best with visual2
    ])
    
    batch_size = 3
    labels = torch.arange(batch_size)
    
    # Audio to Visual direction
    print("1. Raw similarities:")
    print(clip_sims)
    
    print("\n2. After log_softmax along dim=1 (audio to visual):")
    log_prob_a2v = F.log_softmax(clip_sims, dim=1)
    print(log_prob_a2v)
    
    print("\n3. Selecting diagonal elements (correct pairs):")
    losses_a2v = -log_prob_a2v[torch.arange(batch_size), labels]
    print(losses_a2v)
    
    # Visual to Audio direction
    print("\n4. Transposed similarities (visual to audio):")
    print(clip_sims.t())
    
    print("\n5. After log_softmax along dim=1 (visual to audio):")
    log_prob_v2a = F.log_softmax(clip_sims.t(), dim=1)
    print(log_prob_v2a)
    
    print("\n6. Selecting diagonal elements again:")
    losses_v2a = -log_prob_v2a[torch.arange(batch_size), labels]
    print(losses_v2a)
    
    # Final loss
    contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
    print("\n7. Final contrastive loss:", contrastive_loss.item())

if __name__ == "__main__":
    # Test the model
    model = AudioVisualModel()
    
    # Create dummy batch
    batch_size = 4
    frames = torch.randint(0, 65536, (batch_size, 32, 32))
    audio = torch.randn(batch_size, 16331)

    # Test training mode
    loss = model(frames, audio)
    print(f"Training loss: {loss.item()}")
    
    # Test inference mode
    model.eval()
    with torch.no_grad():
        similarities = model(frames, audio)
        print(f"Inference similarities shape: {similarities.shape}")  # Should be (batch_size, Na, 32, 32)
        
    
    
