import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, phoneme_vocab_size, phoneme_embedding_dim,
                 language_vocab_size, language_embedding_dim,
                 speaker_vocab_size, speaker_embedding_dim):
        super(Embeddings, self).__init__()
        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, phoneme_embedding_dim)
        self.language_embedding = nn.Embedding(language_vocab_size, language_embedding_dim)
        self.speaker_embedding = nn.Embedding(speaker_vocab_size, speaker_embedding_dim)
    
    def forward(self, phoneme_ids, language_ids, speaker_ids):
        """
        Args:
            phoneme_ids: Tensor of shape (B, T)
            language_ids: Tensor of shape (B) or (B, T)
            speaker_ids: Tensor of shape (B)
        Returns:
            embeddings: Tensor of shape (B, T, combined_dim)
        """
        # Phoneme embeddings (B, T, phoneme_embedding_dim)
        ph_emb = self.phoneme_embedding(phoneme_ids)
        
        # If language_ids are provided as one value per sample, expand to (B, T)
        if language_ids.dim() == 1:
            language_ids = language_ids.unsqueeze(1).expand(-1, phoneme_ids.size(1))
        lang_emb = self.language_embedding(language_ids)  # (B, T, language_embedding_dim)
        
        # Speaker embeddings (B, speaker_embedding_dim) expanded to (B, T, speaker_embedding_dim)
        spk_emb = self.speaker_embedding(speaker_ids)
        spk_emb = spk_emb.unsqueeze(1).expand(-1, phoneme_ids.size(1), -1)
        
        # Concatenate along the feature dimension
        embeddings = torch.cat([ph_emb, lang_emb, spk_emb], dim=-1)
        return embeddings
