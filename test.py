import model

vae = model.VQVSHX(
    embedding_dim=512,
    hidden_dims=[64],
    in_channels=3,
    learning_rate=1e-3,
    quantizer=model.vq.Quantizer(
        num_embeddings=512,
        embedding_dim=64,
        commitment_cost=0.25,
        sparsity_cost=-1,
        initialize_embedding_b=False,
        embedding_seed=0,
    ),
    scheduler_gamma=0,
    weight_decay=0.0,
    transition_steps=1e5,
)

print(vae)