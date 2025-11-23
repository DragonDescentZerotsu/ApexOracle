import pandas as pd
from pathlib import Path
import numpy as np

current_dir = Path(__file__).parent

peptide_list = pd.read_excel(current_dir / 'raw' / 'Master List Peptides Antimicrobial Activity (1).xlsx')

peptide_dict = {str(name).strip():str(seq).strip() for name, seq in zip(peptide_list['Peptide'].values, peptide_list['Sequence'].values)}

FICI_raw_data_files = [file.name for file in (current_dir/'raw').iterdir() if file.name.startswith('ATCC')]

strain_name_dict = {
    '19606': 'Acinetobacter baumannii ATCC 19606',
    '47085': 'Pseudomonas aeruginosa ATCC 47085'
}

FICI_raw_data = []
strain_data = []
for name in FICI_raw_data_files:
    FICI_data = pd.read_csv(current_dir/'raw'/name).values
    FICI_raw_data.append(FICI_data)
    strain_name = name.split('_')[1]
    strain_data += [strain_name_dict[strain_name]] * len(FICI_data)

FICI_raw_data = np.concatenate(FICI_raw_data, axis=0)

column_names = ['DBAASP_id','antibio_id_or_name','strain_name','AMP_smiles','antibiotic_smiles','FICI']

FICI_precessed_data = []
for data, strain_name in zip(FICI_raw_data, strain_data):
    peptide_id_1, peptide_id_2 = data[0].split('+')
    peptide_id_1 = peptide_id_1.strip()
    peptide_id_2 = peptide_id_2.strip()
    FICI = data[-1]
    pep_seq_1 = peptide_dict.get(peptide_id_1, None)
    pep_seq_2 = peptide_dict.get(peptide_id_2, None)

    if pep_seq_1 is not None and pep_seq_2 is not None:
        FICI_precessed_data.append([peptide_id_1, peptide_id_2, strain_name, pep_seq_1, pep_seq_2, FICI])

df = pd.DataFrame(FICI_precessed_data, columns=column_names)

df.to_csv(current_dir/'processed'/'inhouse_synergy_Evo_pep_seq.csv', index=False)

print(1)