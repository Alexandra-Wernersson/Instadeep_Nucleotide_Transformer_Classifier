###IMPORTS###
import jax
from jax import value_and_grad
import optax
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import pickle


class ClassifierTrainer:
    def __init__(self, model, config):
        self.data_dir = config["DATA PARAMS"]["data_dir"]
        self.result_dir = Path(config["DATA PARAMS"]["result_dir"])
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = config["HYPERPARAMS"]["epochs"]
        self.batch_size = config["HYPERPARAMS"]["batch_size"]
        self.learning_rate = float(config["HYPERPARAMS"]["learning_rate"])

        self.model = model

    def Train_Val_Test(self, train_data, val_data, test_data):
    
        X_train = train_data['sequence'].tolist()
        Y_train = train_data['label'].values

        X_test = test_data['sequence'].tolist()
        Y_test = test_data['label'].values

        X_val = val_data['sequence'].tolist()
        Y_val = val_data['label'].values

        return X_train, Y_train, X_val, Y_val, X_test, Y_test
    
    def get_embeddings(self, sequences, tokenizer, forward_fn_jit, parameters, rng):
        embedding_tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
        embedding_tokens = jnp.asarray(embedding_tokens_ids, dtype=jnp.int32)
    
        outs = forward_fn_jit(parameters, rng, embedding_tokens)
        embeddings = outs["embeddings_12"] 

        return embeddings

    def binary_crossentropy(self, params, rng, input_data, actual, is_training):
        logits = self.model.apply(
            params,
            rng,
            input_data,
            is_training=is_training
        )
        loss = optax.sigmoid_binary_cross_entropy(
            logits=logits,
            labels=actual
        )
        return jnp.mean(loss)

    def TrainModelInBatches(self, train_embeddings, Y_train, val_embeddings, Y_val, params, optimizer, optimizer_state):
    
    
        train_losses = []
        val_losses = []
        train_rng = jax.random.PRNGKey(42)

        best_val = float("inf")
        best_params = None
        patience = 3
        patience_counter = 0

        for epoch in range(int(self.epochs)):
            epoch_train_losses = []
            for start in range(0, len(train_embeddings), int(self.batch_size)):
                end = start + int(self.batch_size)
            # Get embedding batch (not string batch!)
                X_batch = train_embeddings[start:end]  # embeddings, not strings!
                Y_batch = Y_train[start:end]
            
     #       print(f"Batch shapes: X={X_batch.shape}, Y={Y_batch.shape}")
                train_rng, subkey = jax.random.split(train_rng)
                loss, gradients = value_and_grad(self.binary_crossentropy)(params, subkey, X_batch, Y_batch, True)
            
                updates, optimizer_state = optimizer.update(gradients, optimizer_state)
                params = optax.apply_updates(params, updates)
            
                epoch_train_losses.append(loss)

            val_loss = self.binary_crossentropy(params, jax.random.PRNGKey(0), val_embeddings, Y_val, False)
            if val_loss < best_val:
                best_val = val_loss
                best_params = params
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping")
                    break
            # Store average training loss for epoch
            avg_train_loss = jnp.mean(jnp.array(epoch_train_losses))
            train_losses.append(float(avg_train_loss))
            val_losses.append(float(val_loss))
        
        # Print progress
            print(f"Epoch {epoch+1}/{int(self.epochs)} - Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        loss_dict = {
            'train_losses': train_losses,
            'val_losses': val_losses
            }
        
        return params, loss_dict

    
    def save_outs_embeddings_params(
            self,
            embeddings_train,
            embeddings_test,
            params,
            loss_dict,
            prefix
        ):


        # ---- Embeddings ----
        #emb_train_np = jax.device_get(embeddings_train)
        #emb_test_np  = jax.device_get(embeddings_test)

        train_path = self.result_dir / f"{prefix}_train_embeddings.pkl"
        test_path  = self.result_dir / f"{prefix}_test_embeddings.pkl"

        #np.savez_compressed(train_path, embeddings=emb_train_np)
        #np.savez_compressed(test_path, embeddings=emb_test_np)
        with open(train_path, 'wb') as f:
            pickle.dump(embeddings_train, f)
        with open(test_path, 'wb') as f:
            pickle.dump(embeddings_test, f)

        print(f"✓ Saved train embeddings → {train_path}")
        print(f"✓ Saved test embeddings  → {test_path}")

        # ---- Classifier parameters ----
        params_path = self.result_dir / f"{prefix}_classifier_params.pkl"
        #params_np = jax.device_get(params)
        with open(params_path, "wb") as f:
            pickle.dump(params, f)
        print(f"✓ Saved classifier params → {params_path}")

        # ---- Loss curves ----
        loss_path = self.result_dir / f"{prefix}_losses.pkl"
        with open(loss_path, "wb") as f:
           pickle.dump(loss_dict, f)
        print(f"✓ Saved losses → {loss_path}")
