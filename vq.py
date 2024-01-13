import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn


class Quantizer(eqx.Module):
    num_embeddings: int = 512
    embedding_dim: int = 64
    commitment_cost: float = 0.25
    sparsity_cost: float = 1.0
    initialize_embedding_b: bool = True
    embedding_seed: int = 0
    
    embedding: nn.Embedding
    B: float
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, sparsity_cost, initialize_embedding_b, embedding_seed):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.sparsity_cost = sparsity_cost
        self.initialize_embedding_b = initialize_embedding_b
        self.embedding_seed = embedding_seed
        
        self.B = 1 / ((self.num_embeddings ** (1 / self.embedding_dim)) - 1)
        
        if self.initialize_embedding_b:
            self.embedding = nn.Embedding(weight=jax.random.uniform(
                key=jax.random.PRNGKey(embedding_seed),
                shape=(1, self.embedding_dim),
                minval=-self.B,
                maxval=self.B,
            ))
        else:
            self.embedding = nn.Embedding(weight=jax.random.uniform(
                key=jax.random.PRNGKey(embedding_seed),
                shape=(self.num_embeddings, self.embedding_dim),
                minval=-1,
                maxval=1,
            ))
    
    def mse_loss(self, x, y):
        return jnp.mean(jnp.square(x - y))
    
    def __call__(self, latents: jnp.ndarray) -> dict:
        # [B x D x H x W] -> [B x H x W x D]
        latents = latents.transpose((0, 2, 3, 1))
        
        # [BHW x D]
        flat_latents = latents.reshape(-1, self.embedding_dim)
        
        # Babai estimate
        babai_estimate = jnp.round(
            jnp.multiply(flat_latents, 1 / self.embedding.weight)
        )
        
        # Quantize latents
        quantized_latents_flat = jnp.multiply(self.embedding.weight, babai_estimate)
        quantized_latents = quantized_latents_flat.reshape(latents.shape) # [BHW x D] -> [B x H x W x D]
        
        # Compute LQ loss
        commitment_loss = self.mse_loss(quantized_latents, jax.lax.stop_gradient(latents))
        embedding_loss = self.mse_loss(jax.lax.stop_gradient(quantized_latents), latents)
        
        size_loss = -jnp.sum(jnp.abs(self.embedding.weight))
        lq_loss = (
            embedding_loss
            + self.commitment_cost * commitment_loss
            + self.sparsity_cost * size_loss
        )
        
        # Add the residue back to the quantized latents
        quantized_latents = latents + jax.lax.stop_gradient(
            jnp.subtract(latents, quantized_latents)
        )
        
        return {
            'vq_loss': lq_loss,
            'quantized': quantized_latents.transpose((0, 3, 1, 2)),
            'quantized_flat': quantized_latents_flat,
        }


class ResidualLayer(eqx.Module):
    resblock: nn.Sequential
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.resblock = nn.Sequential([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                use_bias=False,
                key=jax.random.PRNGKey(0)
            ),
            nn.Lambda(jax.nn.relu),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                use_bias=False,
                key=jax.random.PRNGKey(0)
            )
        ])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x + self.resblock(x)