import torch
import torch.nn as nn

class AutoencoderCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_seq_length):
        super(AutoencoderCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden_dim, max_seq_length, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)
        return x
