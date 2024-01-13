import jax
import optax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import vq

from typing import List


class VQVSHX(eqx.Module):
    embedding_dim: int = 64
    hidden_dims: List[int] = None
    in_channels: int = None
    learning_rate: float = 1e-3
    quantizer: vq.Quantizer = None
    scheduler_gamma: float = 0.0
    weight_decay: float = 0.0
    transition_steps: int = 1e5
    
    _encoder: nn.Sequential
    _decoder: nn.Sequential
    
    optimizer: optax.GradientTransformation
    scheduler: optax.Schedule
    
    def __init__(self, embedding_dim, hidden_dims, in_channels, learning_rate, quantizer, scheduler_gamma, weight_decay, transition_steps):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        self.learning_rate = learning_rate
        self.quantizer = quantizer
        self.scheduler_gamma = scheduler_gamma
        self.weight_decay = weight_decay
        self.transition_steps = transition_steps
        
        modules = []
        out_channels = in_channels
        hidden_dims = hidden_dims
        
        # Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential([
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=h_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        key=jax.random.PRNGKey(0)
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ])
            )
            in_channels = h_dim
        
        modules.append(
            nn.Sequential([
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    key=jax.random.PRNGKey(0)
                ),
                nn.Lambda(jax.nn.leaky_relu),
            ])
        )
        
        for _ in range(2):
            modules.append(vq.ResidualLayer(in_channels, in_channels))
        modules.append(nn.Lambda(jax.nn.leaky_relu))
        
        modules.append(
            nn.Sequential(
                nn.Sequential([
                    nn.Conv2d(
                        in_channels,
                        embedding_dim,
                        kernel_size=1,
                        stride=1,
                        key=jax.random.PRNGKey(0)
                    )
                ])
            )
        )
        
        self._encoder = nn.Sequential(modules)
        
        # Decoder
        modules = []
        
        modules.append(
            nn.Sequential([
                nn.Conv2d(
                    embedding_dim,
                    hidden_dims[-1],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    key=jax.random.PRNGKey(0)
                ),
                nn.Lambda(jax.nn.leaky_relu),
            ])
        )
        
        for _ in range(2):
            modules.append(vq.ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        modules.append(nn.Lambda(jax.nn.leaky_relu))
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential([
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        key=jax.random.PRNGKey(0),
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ])
            )
        
        modules.append(
            nn.Sequential([
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    key=jax.random.PRNGKey(0),
                ),
                nn.Lambda(jax.nn.leaky_relu),
            ])
        )
        
        self._decoder = nn.Sequential(modules)
        modules = []
        
        self.configure_optimizers()
    
    def configure_optimizers(self):
        self.optimizer = optax.adamw(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = optax.exponential_decay(
            init_value=self.learning_rate,
            transition_steps=self.transition_steps,
            decay_rate=self.scheduler_gamma,
        )