import concurrent.futures
import numpy as np
from rdkit import Chem
from pathlib import Path
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import pandas as pd
from tqdm import tqdm

current_dir = Path(__file__).parent

def compute_fingerprint(smile):
    """Convert a SMILES string to a Morgan fingerprint (radius=2, 2048 bits)."""
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    fp = generator.GetFingerprint(mol)
    return fp


def compute_similarity_pair(args):
    """
    Given a tuple (i, j, fp_i, fp_j), compute the Tanimoto similarity.
    Returns a tuple (i, j, similarity).
    """
    i, j, fp_i, fp_j = args
    sim = DataStructs.TanimotoSimilarity(fp_i, fp_j)
    return (i, j, sim)


def main(smiles_list):
    print(f' Total length of sequences: {len(smiles_list)}')

    # 1. Compute fingerprints in parallel.
    print('\n Generating Morgan Fingerprints ...')
    with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
        fingerprints = list(executor.map(compute_fingerprint, smiles_list))

    # Ensure all SMILES strings were parsed correctly.
    if any(fp is None for fp in fingerprints):
        raise ValueError("One or more SMILES strings could not be parsed.")

    n = len(fingerprints)

    # 2. Create tasks only for pairs in the upper triangular portion (i < j).
    tasks = []
    for i in range(n):
        for j in range(i + 1, n):
            tasks.append((i, j, fingerprints[i], fingerprints[j]))

    # Preallocate an empty similarity matrix.
    similarity_matrix = np.zeros((n, n))

    # 3. Compute similarity in parallel for each pair.
    print('\n Calculating similarity matrix ...')
    with concurrent.futures.ProcessPoolExecutor(max_workers=128) as executor:
        for i, j, sim in tqdm(executor.map(compute_similarity_pair, tasks), desc=' Comparing Morgan Fingerprints', total=len(tasks)):
            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim  # Mirror the result since similarity is symmetric.

    # Optionally, set the diagonal to 1.0 (self-similarity).
    np.fill_diagonal(similarity_matrix, 1.0)

    return similarity_matrix

def get_smiles_list(path_to_merged_smiles: Path):
    df = pd.read_csv(path_to_merged_smiles)
    smiles_list = df['SMILES'].values
    DBAASP_ids = df['DBAASP_id'].values
    return DBAASP_ids, smiles_list

if __name__ == '__main__':
    # Example list of SMILES strings.
    # smiles_list = [
    #     "CCO",  # Ethanol
    #     "CCN",  # Ethylamine
    #     "CCC",  # Propane
    #     "c1ccccc1",  # Benzene
    #     "C1=CC=CN=C1"  # Pyridine
    # ]
    merged_smiles_path = current_dir / 'Data' / 'inhouse_DBAASP_id_SMILES_merged.csv'
    DBAASP_ids, smiles_list = get_smiles_list(merged_smiles_path)

    sim_matrix = main(smiles_list)
    print("Similarity Matrix:")
    print(sim_matrix)

    np.save(current_dir / 'Data' / "inhouse_DBAASP_peptides_sim_matrix.npy", sim_matrix)