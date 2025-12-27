import os
import ast
#try:
#    import nucleotide_transformer
#except:
#    pip install numpy==1.23.5
#    pip install git+https://github.com/instadeepai/nucleotide-transformer@main |tail -n 1
import nucleotide_transformer


import sys
import haiku as hk
import jax
import torch
import jax.numpy as jnp
import optax
from nucleotide_transformer.pretrained import get_pretrained_model
import pickle
from datetime import datetime
import configparser

from data_utils import FastPromoterDataset, RealPromoterDataset
from training_utils import ClassifierTrainer
from Classifier import PromoterClassifierNet
if __name__ == "__main__":
    
    # === 1. Load config ===
    #Either load data, create it or config with data and model
    if len(sys.argv) < 2:
       print("Usage: python train_cosmo.py <config_file>")
       sys.exit(1)
    config_path = sys.argv[1]
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] | Reading config file: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)

    # Load data or generate
    data_params = config['DATA PARAMS']
    dataset = FastPromoterDataset(data_dir=data_params['data_dir'])
    print(data_params['load_data'])
    if data_params['load_data'] == True:
        #data = pd.read_csv(data_params['data_dir'])
        data = dataset.load_data()
    else:
        data = dataset.create_dataset(n_samples=data_params['n_samples'], seq_length=int(data_params['seq_length']))
        dataset.save_dataset(data, prefix=data_params['data_prefix'])

    # Load the model
    model_params = config["MODEL PARAMS"]
    model_name = model_params["model_name"]
    
    parameters, forward_fn, tokenizer, _ = get_pretrained_model(
        model_name=model_params["model_name"],
        compute_dtype=jnp.float32,  
        param_dtype=jnp.float32,
        output_dtype=jnp.float32,
        embeddings_layers_to_save=ast.literal_eval(
        model_params["embedding_layers"]
        ), 
        attention_maps_to_save=ast.literal_eval(
        model_params["attention_maps"]
        ),
        max_positions=int(model_params["max_positions"]),
    )

    rng = jax.random.PRNGKey(0)

    sequences = data["train"]["sequence"]

    tokens_ids = [b[1] for b in tokenizer.batch_tokenize(sequences)]
    tokens_str = [b[0] for b in tokenizer.batch_tokenize(sequences)]
    tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)

    forward_fn = hk.transform(forward_fn)
    forward_fn_jit = jax.jit(forward_fn.apply)

    # Run inference
    outs = forward_fn_jit(parameters, rng, tokens)

    # Extract X/Y_train, X/Y_val, X/Y_test
    model = hk.transform(PromoterClassifierNet)
    trainer = ClassifierTrainer(model, config)
    train_data = data['train']  
    val_data = data['val']
    test_data = data['test']

    X_train, Y_train, X_val, Y_val, X_test, Y_test = trainer.Train_Val_Test(train_data, val_data, test_data)
    
    train_embeddings = trainer.get_embeddings(
        X_train, tokenizer, forward_fn_jit, parameters, rng
    )

    val_embeddings = trainer.get_embeddings(
        X_val, tokenizer, forward_fn_jit, parameters, rng
    )

    test_embeddings = trainer.get_embeddings(   
        X_test, tokenizer, forward_fn_jit, parameters, rng
    )
    # Train model
    hyperparams = config["HYPERPARAMS"]
    model = hk.transform(PromoterClassifierNet)
    params = model.init(rng, outs["embeddings_12"])
    optimizer = optax.adam(learning_rate=float(hyperparams['learning_rate']))
    optimizer_state = optimizer.init(params)
    final_params, loss_dict = trainer.TrainModelInBatches(train_embeddings, 
                                                          Y_train, 
                                                          test_embeddings, 
                                                          Y_test,
                                                          params,
                                                          optimizer, 
                                                          optimizer_state) 
    trainer.save_outs_embeddings_params(
        train_embeddings,
        test_embeddings,
        final_params,
        loss_dict,
        data_params["data_prefix"]
    )
    print("Training done and results saved!")    
