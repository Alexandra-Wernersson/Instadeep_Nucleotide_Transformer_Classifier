###IMPORTS###
import jax
from jax import value_and_grad
import optax
from tqdm import tqdm
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import pickle
import haiku as hk

class PromoterClassifier(hk.Module):
    def __init__(self, hidden_dim=128, dropout_rate=0.3):
        super().__init__(name="PromoterClassifier")
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

    def __call__(self, embeddings_input, is_training: bool):
        # CLS token
        x = embeddings_input[:, 0, :]  # [batch, embed_dim]

        # ---- MLP ----
        x = hk.Linear(self.hidden_dim, name="dense1")(x)
        x = jax.nn.relu(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)

        x = hk.Linear(64, name="dense2")(x)
        x = jax.nn.relu(x)
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)

        # ---- Output ----
        logits = hk.Linear(1, name="output")(x)
        return logits

def PromoterClassifierNet(x, is_training: bool):
    model = PromoterClassifier()
    return model(x, is_training)

