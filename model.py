import jax
import optax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as nn
import vq

from typing import Any, List


class VQVSHX(eqx.Module):
    _encoder: nn.Sequential
    _decoder: nn.Sequential
    _quantizer: vq.Quantizer
    
    learning_rate: float = 1e-3
    scheduler_gamma: float = 0.0
    weight_decay: float = 0.0
    
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [64],
        in_channels: int = 3,
        learning_rate: float = 1e-3,
        quantizer: vq.Quantizer = None,
        scheduler_gamma: float = 0.0,
        weight_decay: float = 0.0,
    ):
        # super(VQVSHX, self).__init__()
        
        self.learning_rate = learning_rate
        self.scheduler_gamma = scheduler_gamma
        self.weight_decay = weight_decay
        self._quantizer = quantizer
        
        out_channels = in_channels
        
        self._create_encoder(in_channels, hidden_dims, embedding_dim)
        self._create_decoder(out_channels, hidden_dims, embedding_dim)
    
    def _create_encoder(self, in_channels: int, hidden_dims: List[int], embedding_dim: int):
        layers = []
        # hidden_dims = hidden_dims
        
        for h_dim in hidden_dims:
            layers.append(
                nn.Sequential([
                    nn.Conv2d(
                        in_channels,
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
        
        layers.append(
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
            layers.append(vq.ResidualLayer(in_channels, in_channels))
        layers.append(nn.Lambda(jax.nn.leaky_relu))
        
        layers.append(
            nn.Sequential([
                nn.Conv2d(
                    in_channels,
                    embedding_dim,
                    kernel_size=1,
                    stride=1,
                    key=jax.random.PRNGKey(0)
                ),
                nn.Lambda(jax.nn.leaky_relu),
            ])
        )
        
        self._encoder = nn.Sequential(layers)
        layers = []
    
    def _create_decoder(self, out_channels: int, hidden_dims: List[int], embedding_dim: int):
        layers = []
        hidden_dims = hidden_dims
        
        layers.append(
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
            layers.append(vq.ResidualLayer(hidden_dims[-1], hidden_dims[-1]))
        layers.append(nn.Lambda(jax.nn.leaky_relu))
        
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            layers.append(
                nn.Sequential([
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        key=jax.random.PRNGKey(0)
                    ),
                    nn.Lambda(jax.nn.leaky_relu),
                ])
            )
        
        layers.append(
            nn.Sequential([
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    key=jax.random.PRNGKey(0)
                ),
                nn.Lambda(jax.nn.leaky_relu),
            ])
        )
        
        self._decoder = nn.Sequential(layers)
        layers = []
    
    def __call__(self, x):
        z = self._encoder(x)
        vq_output = self._quantizer(z)
        x_recon = self._decoder(vq_output['quantized'])
        
        return {
            'data_recon': x_recon,
            **{k: v for k, v in vq_output.items() if 'vq' in k}
        }