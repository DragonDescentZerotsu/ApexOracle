import pandas as pd
import selfies as sf
from transformers import AutoTokenizer
from tqdm import tqdm


def convert_smiles_to_selfies_tokens(input_csv: str,
                                     output_csv: str,
                                     smiles_col: str = 'SMILES',
                                     model_name: str = 'ibm-research/materials.selfies-ted',
                                     max_length: int = 1024):
    """
    Reads a CSV with a SMILES column, converts SMILES to SELFIES,
    tokenizes with the specified HF tokenizer, filters out sequences
    > max_length tokens or containing unknown tokens, and writes
    the result back to a new CSV (original SMILES column replaced).

    Args:
        input_csv: Path to input CSV file.
        output_csv: Path to save the filtered and tokenized CSV.
        smiles_col: Name of the SMILES column in the CSV.
        model_name: Hugging Face model name for SELFIES tokenizer.
        max_length: Maximum number of tokens allowed.
    """
    # Load data
    df = pd.read_csv(input_csv)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    unk_id = tokenizer.unk_token_id

    token_lists = []
    valid_indices = []

    for idx, smiles in tqdm(enumerate(df[smiles_col].astype(str)), desc='tokenizing SELFIES', total=len(df)):
        try:
            # Convert to SELFIES and ensure separation of bracket tokens
            selfies_str = sf.encoder(smiles)
            selfies_str = selfies_str.replace('][', '] [')
        except Exception as e:
            # Skip invalid SMILES that cannot be encoded
            print(f"Warning: could not convert SMILES at row {idx}: {e}")
            continue

        # Tokenize SELFIES string
        encoding = tokenizer(selfies_str, add_special_tokens=True)
        ids = encoding['input_ids']

        # Filter sequences
        if len(ids) <= max_length and unk_id not in ids:
            token_lists.append(ids)
            valid_indices.append(idx)
        else:
            # Skip sequences too long or containing unknowns
            continue

    # Subset dataframe to only valid rows
    filtered_df = df.iloc[valid_indices].copy()
    # Replace SMILES column with token lists
    filtered_df[smiles_col] = token_lists

    # Save to output CSV
    print(f'writing {len(filtered_df)} tokens to {output_csv}')
    filtered_df.to_csv(output_csv, index=False)
    print(f'Wrote {len(filtered_df)} tokens to {output_csv}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert SMILES column in CSV to SELFIES tokens and filter.'
    )
    parser.add_argument('--input_csv', type=str, help='Input CSV file path', default='/data2/tianang/projects/Synergy/DataPrepare/Data/DBAASP_inhouse_AMP_SMILES_MIC_Evo.csv')
    parser.add_argument('--output_csv', type=str, help='Output CSV file path', default='/data2/tianang/projects/Synergy/DataPrepare/Data/DBAASP_inhouse_AMP_SELFIES_token_MIC_Evo.csv')
    parser.add_argument('--smiles_col', type=str, default='SMILES',
                        help='Name of the SMILES column')
    parser.add_argument('--model_name', type=str,
                        default='ibm-research/materials.selfies-ted',
                        help='HF tokenizer model name')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum number of tokens')

    args = parser.parse_args()
    convert_smiles_to_selfies_tokens(
        args.input_csv,
        args.output_csv,
        smiles_col=args.smiles_col,
        model_name=args.model_name,
        max_length=args.max_length
    )
