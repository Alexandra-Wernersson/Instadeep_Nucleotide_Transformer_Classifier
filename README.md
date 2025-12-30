# Instadeep_Nucleotide_Transformer_Classifier

This project uses the pre-trained **Nucleotide Transformer** model from **InstaDeep AI**  
(https://github.com/instadeepai/nucleotide-transformer) to extract DNA sequence embeddings.

Using these embeddings, a neural network classifier is trained to distinguish **promoter**
from **non-promoter** DNA sequences.

The promoter sequence dataset is downloaded and processed automatically during training.

The trained classifier achieves approximately **85% classification accuracy** on a
test set.

Model performance and interpretability results — including **training/validation loss curves,
attention maps, precision-recall curves, ROC curves, and a classification report** —
are visualized in `show_results_final.ipynb`.

---

## Project Workflow

### Step 1: Data configuration and download
Specify the number of samples and sequence length in:

`examples/example_config.ini`.

Then download and preprocess the data using:

`python data_utils.py examples/example_config.ini`.

Alternatively, the data will be downloaded automatically when running the training script.
---

### Step 2: Train the classifier

Run:

`python train.py examples/example_config.ini`


This will:
- Extract embeddings using the Nucleotide Transformer
- Train the promoter classifier
- Save embeddings, trained model parameters, and logs to a `/results` directory

---

### Step 3: Evaluate and visualize results

Open the notebook:

`show_results_final.ipynb`.

This notebook loads the saved results and visualizes:
- Training and validation loss curves
- Attention maps from the transformer model
- Precision–Recall and ROC curves
- Test-set classification metrics


## Results Summary
- Accuracy: ~85%
- Macro F1-score: ~0.85
- Strong precision-recall balance across classes
