import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log

# Encoder Block for downsampling
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation="relu"):
        super().__init__()
        # Define a convolutional block with specified input/output channels and time steps
        self.conv = conv_block(in_c, out_c, time_steps=time_steps, activation=activation, embedding_dims=out_c)
        # Define a max pooling layer to downsample the input
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, time=None):
        # Apply convolutional block
        x = self.conv(inputs, time)
        # Apply max poolingcd 
        p = self.pool(x)
        return x, p

# Decoder Block for upsampling
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps, activation="relu"):
        super().__init__()
        # Define a transposed convolutional layer for upsampling
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        # Define a convolutional block to process the concatenated feature maps
        self.conv = conv_block(out_c + out_c, out_c, time_steps=time_steps, activation=activation, embedding_dims=out_c)

    def forward(self, inputs, skip, time=None):
        # Apply transposed convolution for upsampling
        x = self.up(inputs)
        # Concatenate the skip connection with the upsampled input
        x = torch.cat([x, skip], axis=1)
        # Apply convolutional block
        x = self.conv(x, time)
        return x

# Gamma Encoding for timestep embedding
class GammaEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU()

    def forward(self, noise_level):
        if isinstance(noise_level, int):
            noise_level = torch.tensor([noise_level], dtype=torch.float32, device='cpu')
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return self.act(self.linear(encoding))

# Double Convolution Block
class conv_block(nn.Module):
    def __init__(self, in_c, out_c, time_steps=1000, activation="relu", embedding_dims=None):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        # Batch normalization for the first convolutional layer
        self.bn1 = nn.BatchNorm2d(out_c)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        # Batch normalization for the second convolutional layer
        self.bn2 = nn.BatchNorm2d(out_c)
        # Set embedding dimensions
        self.embedding_dims = embedding_dims if embedding_dims else out_c
        # Define a gamma encoding layer for timestep embedding
        self.embedding = GammaEncoding(self.embedding_dims)
        # Activation function
        self.act = nn.ReLU() if activation == "relu" else nn.SiLU()

    def forward(self, inputs, time=None):
        # Get time embeddings
        time_embedding = self.embedding(time).view(-1, self.embedding_dims, 1, 1)
        # Apply the first convolutional layer and activation
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act(x)
        # Apply the second convolutional layer and activation
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        # Add the time embedding
        x = x + time_embedding
        return x

# Define the Attention Block
class AttnBlock(nn.Module):
    def __init__(self, embedding_dims, num_heads=4):
        super().__init__()
        self.embedding_dims = embedding_dims
        # Layer normalization
        self.ln = nn.LayerNorm(embedding_dims)
        # Multi-head self-attention mechanism
        self.mhsa = MultiHeadSelfAttention(embedding_dims=embedding_dims, num_heads=num_heads)
        # Feed-forward network with layer normalization and activation
        self.ff = nn.Sequential(
            nn.LayerNorm(self.embedding_dims),
            nn.Linear(self.embedding_dims, self.embedding_dims),
            nn.GELU(),
            nn.Linear(self.embedding_dims, self.embedding_dims),
        )

    def forward(self, x):
        bs, c, sz, _ = x.shape
        # Reshape and swap axes for attention mechanism
        x = x.view(-1, self.embedding_dims, sz * sz).swapaxes(1, 2)
        x_ln = self.ln(x)
        # Apply multi-head self-attention
        _, attention_value = self.mhsa(x_ln, x_ln, x_ln)
        # Residual connection
        attention_value = attention_value + x
        # Apply feed-forward network
        attention_value = self.ff(attention_value) + attention_value
        # Reshape and swap axes back
        return attention_value.swapaxes(2, 1).view(-1, c, sz, sz)

# Define the Multi-Head Self-Attention block
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dims, num_heads=4):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        assert self.embedding_dims % self.num_heads == 0, f"{self.embedding_dims} not divisible by {self.num_heads}"
        self.head_dim = self.embedding_dims // self.num_heads
        # Linear layers for query, key, and value
        self.wq = nn.Linear(self.head_dim, self.head_dim)
        self.wk = nn.Linear(self.head_dim, self.head_dim)
        self.wv = nn.Linear(self.head_dim, self.head_dim)
        # Linear layer for output
        self.wo = nn.Linear(self.embedding_dims, self.embedding_dims)

    def attention(self, q, k, v):
        # Compute attention weights and output
        attn_weights = F.softmax((q @ k.transpose(-1, -2)) / self.head_dim ** 0.5, dim=-1)
        return attn_weights, attn_weights @ v

    def forward(self, q, k, v):
        bs, img_sz, c = q.shape
        # Reshape and transpose for multi-head attention
        q = q.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, img_sz, self.num_heads, self.head_dim).transpose(1, 2)
        # Apply linear layers for query, key, and value
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        # Compute attention
        attn_weights, o = self.attention(q, k, v)
        # Reshape and apply output linear layer
        o = o.transpose(1, 2).contiguous().view(bs, img_sz, self.embedding_dims)
        o = self.wo(o)
        return attn_weights, o

# Define the U-Net with Self-Attention
class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, time_steps=512):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_steps = time_steps

        # Define encoder blocks
        self.e1 = encoder_block(self.input_channels, 64, time_steps=self.time_steps)
        self.e2 = encoder_block(64, 128, time_steps=self.time_steps)
        self.e3 = encoder_block(128, 256, time_steps=self.time_steps)
        # Add attention block to the encoder
        self.da3 = AttnBlock(256)
        self.e4 = encoder_block(256, 512, time_steps=self.time_steps)
        self.da4 = AttnBlock(512)

        # Define bottleneck layer
        self.b = conv_block(512, 1024, time_steps=self.time_steps)
        self.ba1 = AttnBlock(1024)
        # Define decoder blocks
        self.d1 = decoder_block(1024, 512, time_steps=self.time_steps)
        self.ua1 = AttnBlock(512)
        self.d2 = decoder_block(512, 256, time_steps=self.time_steps)
        self.ua2 = AttnBlock(256)
        self.d3 = decoder_block(256, 128, time_steps=self.time_steps)
        self.d4 = decoder_block(128, 64, time_steps=self.time_steps)
        # Define output layer
        self.outputs = nn.Conv2d(64, self.output_channels, kernel_size=1, padding=0)

    def forward(self, inputs, t=None):
        # Forward pass through the encoder blocks
        s1, p1 = self.e1(inputs, t)
        s2, p2 = self.e2(p1, t)
        s3, p3 = self.e3(p2, t)
        p3 = self.da3(p3)
        s4, p4 = self.e4(p3, t)
        p4 = self.da4(p4)
        # Forward pass through the bottleneck layer
        b = self.b(p4, t)
        b = self.ba1(b)
        # Forward pass through the decoder blocks
        d1 = self.d1(b, s4, t)
        d1 = self.ua1(d1)
        d2 = self.d2(d1, s3, t)
        d2 = self.ua2(d2)
        d3 = self.d3(d2, s2, t)
        d4 = self.d4(d3, s1, t)
        # Get the final output
        outputs = self.outputs(d4)
        return outputs

# Define the Diffusion Model
class DiffusionModel(nn.Module):
    def __init__(self, time_steps, beta_start=10e-4, beta_end=0.02, image_dims=(3, 128, 128)):
        super().__init__()
        self.time_steps = time_steps
        self.image_dims = image_dims
        c, h, w = self.image_dims
        self.img_size, self.input_channels = h, c
        # Define betas for the diffusion process
        self.betas = torch.linspace(beta_start, beta_end, self.time_steps)
        # Define alphas and cumulative product of alphas
        self.alphas = 1 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=-1)
        # Define the U-Net model
        self.model = UNet(input_channels=2*c, output_channels=c, time_steps=self.time_steps)

    def add_noise(self, x, ts):
        # Add noise to the input images
        noise = torch.randn_like(x)
        noised_examples = []
        for i, t in enumerate(ts):
            alpha_hat_t = self.alpha_hats[t]
            noised_examples.append(torch.sqrt(alpha_hat_t) * x[i] + torch.sqrt(1 - alpha_hat_t) * noise[i])
        return torch.stack(noised_examples), noise

    def forward(self, x, t):
        # Forward pass through the U-Net model
        return self.model(x, t)
