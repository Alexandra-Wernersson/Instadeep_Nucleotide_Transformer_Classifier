import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

class FastPromoterDataset:
    """
    Fast promoter dataset generator
    Creates biologically realistic sequences instantly
    """

    def __init__(self, data_dir='./data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        np.random.seed(42)

    def create_dataset(self,
                      n_samples=1000,
                      seq_length=50,  # Short sequences like InstaDeep example
                      split_ratio=(0.7, 0.15, 0.15)):
        """
        Create a complete train/val/test dataset instantly

        Args:
            n_samples: Total number of samples (half promoters, half non-promoters)
            seq_length: Length of each sequence (default 200bp)
            split_ratio: (train, val, test) ratios

        Returns:
            Dictionary with train/val/test DataFrames
        """
        print(f"Creating dataset: {n_samples} samples, {seq_length}bp each")

        # Generate sequences
        sequences = []
        labels = []
        sequence_types = []

        # Half promoters
        n_promoters = int(n_samples) // 2
        for i in range(n_promoters):
            seq = self._generate_promoter_sequence(int(seq_length))
            sequences.append(seq)
            labels.append(1)
            sequence_types.append('promoter')

        # Half non-promoters
        n_non_promoters = int(n_samples) - n_promoters
        for i in range(n_non_promoters):
            seq = self._generate_non_promoter_sequence(int(seq_length))
            sequences.append(seq)
            labels.append(0)
            sequence_types.append('non_promoter')

        # Create DataFrame
        df = pd.DataFrame({
            'sequence': sequences,
            'label': labels,
            'type': sequence_types,
            'length': [len(s) for s in sequences]
        })

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train/val/test
        n_train = int(len(df) * split_ratio[0])
        n_val = int(len(df) * split_ratio[1])

        train = df[:n_train]
        val = df[n_train:n_train+n_val]
        test = df[n_train+n_val:]

        print(f"\n✓ Dataset created:")
        print(f"  Train: {len(train)} ({train['label'].sum()} promoters)")
        print(f"  Val:   {len(val)} ({val['label'].sum()} promoters)")
        print(f"  Test:  {len(test)} ({test['label'].sum()} promoters)")

        return {
            'train': train,
            'val': val,
            'test': test,
            'full': df
        }

    def _generate_promoter_sequence(self, length=200):
        """
        Generate a realistic promoter sequence with known motifs

        Promoters typically contain:
        - TATA box (~25-30bp upstream of TSS)
        - Initiator element (Inr) at TSS
        - CpG islands (high GC content)
        - CAAT box
        """
        parts = []
        remaining = length

        # Upstream region with high GC content (CpG island characteristic)
        gc_region_len = min(80, remaining // 3)
        parts.append(self._generate_gc_rich_region(gc_region_len))
        remaining -= gc_region_len

        # CAAT box (70% of promoters have this)
        if np.random.random() > 0.3:
            caat_variants = ['CCAAT', 'GGCCAATCT', 'ATTGG']
            parts.append(np.random.choice(caat_variants))
            remaining -= len(parts[-1])

        # Spacer
        spacer_len = min(30, remaining // 3)
        parts.append(''.join(np.random.choice(['A','T','G','C'], spacer_len)))
        remaining -= spacer_len

        # TATA box (40% of human promoters have this)
        if np.random.random() > 0.6:
            tata_variants = ['TATAAA', 'TATAA', 'TATTAA', 'TATAAAT']
            parts.append(np.random.choice(tata_variants))
            remaining -= len(parts[-1])

        # Spacer
        spacer_len = min(15, remaining // 2)
        parts.append(''.join(np.random.choice(['A','T','G','C'], spacer_len)))
        remaining -= spacer_len

        # Initiator element (Inr) at TSS
        if np.random.random() > 0.3:
            inr_variants = ['YYANWYY', 'TCAGT', 'TCACT', 'CTCAGT']
            inr = np.random.choice(inr_variants)
            # Replace ambiguous bases
            inr = inr.replace('Y', np.random.choice(['C', 'T']))
            inr = inr.replace('W', np.random.choice(['A', 'T']))
            inr = inr.replace('N', np.random.choice(['A','T','G','C']))
            parts.append(inr)
            remaining -= len(inr)

        # Downstream region
        if remaining > 0:
            parts.append(''.join(np.random.choice(['A','T','G','C'], remaining)))

        sequence = ''.join(parts)

        # Ensure exactly the right length
        if len(sequence) > length:
            sequence = sequence[:length]
        elif len(sequence) < length:
            sequence += ''.join(np.random.choice(['A','T','G','C'], length - len(sequence)))

        # IMPORTANT: Return uppercase for tokenizer compatibility
        return sequence.upper()

    def _generate_non_promoter_sequence(self, length=200):
        """
        Generate a non-promoter sequence
        - Random sequence OR
        - Coding sequence (exon) OR
        - Intergenic region
        """
        seq_type = np.random.choice(['random', 'coding', 'intergenic'])

        if seq_type == 'random':
            # Truly random
            return ''.join(np.random.choice(['A','T','G','C'], length)).upper()  # UPPERCASE

        elif seq_type == 'coding':
            # Coding sequence characteristics:
            # - Must be in-frame (multiples of 3)
            # - More structured
            # - Lower GC content than promoters
            codons = []
            for _ in range(length // 3):
                codon = ''.join(np.random.choice(['A','T','G','C'], 3))
                codons.append(codon)
            seq = ''.join(codons)
            # Pad if needed
            if len(seq) < length:
                seq += ''.join(np.random.choice(['A','T','G','C'], length - len(seq)))
            return seq[:length].upper()  # UPPERCASE for tokenizer

        else:  # intergenic
            # Intergenic regions: more AT-rich
            bases = np.random.choice(['A','T','G','C'], length, p=[0.35, 0.35, 0.15, 0.15])
            return ''.join(bases).upper()  # UPPERCASE for tokenizer

    def _generate_gc_rich_region(self, length):
        """Generate GC-rich region (CpG island)"""
        bases = np.random.choice(['A','T','G','C'], length, p=[0.15, 0.15, 0.35, 0.35])
        return ''.join(bases)  # Already uppercase


    def save_dataset(self, dataset_dict, prefix='promoter_dataset'):
        """Save dataset splits to CSV files"""
        for split_name, df in dataset_dict.items():
            if split_name == 'full':
                continue
            filepath = self.data_dir / f"{prefix}_{split_name}.csv"
            df.to_csv(filepath, index=False)
            print(f"✓ Saved {split_name} to {filepath}")

    def load_dataset(self, prefix='promoter_dataset'):
        """Load dataset from saved files"""
        dataset = {}
        for split in ['train', 'val', 'test']:
            filepath = self.data_dir / f"{prefix}_{split}.csv"
            if filepath.exists():
                dataset[split] = pd.read_csv(filepath)
                print(f"✓ Loaded {split}: {len(dataset[split])} samples")
        return dataset

    


class RealPromoterDataset:
    """
    Load real promoter sequences from public databases
    Uses pre-downloaded data or curated lists
    """

    @staticmethod
    def load_from_gencode():
        """
        Instructions to download real GENCODE promoters
        This is the gold standard for human gene annotations
        """
        print("="*60)
        print("Loading Real Promoter Data from GENCODE")
        print("="*60)
        print("\nTo get real promoter sequences:")
        print("\n1. Download GENCODE GTF:")
        print("   wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz")

        print("\n2. Extract TSS positions:")
        print("   This will give you exact transcription start sites")

        print("\n3. Use the provided script to extract sequences")
        print("="*60)

        # For now, return None but provide instructions
        return None


