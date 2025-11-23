import os
os.environ["OMP_NUM_THREADS"] = "3"

import concurrent.futures
import numpy as np
from rdkit import Chem
from pathlib import Path
from rdkit.Chem import AllChem, DataStructs, rdFingerprintGenerator
import pandas as pd
from tqdm import tqdm
import selfies as sf
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors  # 仅用于颜色处理
from matplotlib.colors import LinearSegmentedColormap

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


def main(DBAASP_smiles_list, generated_smiles_list):
    print(f' Total length of DBAASP sequences: {len(DBAASP_smiles_list)}')
    print(f' Total length of generated sequences: {len(generated_smiles_list)}')

    # 1. Compute fingerprints in parallel.
    print('\n Generating Morgan Fingerprints ...')
    with Pool(processes=cpu_count()) as p:
        DBAASP_fps = list(tqdm(p.imap(compute_fingerprint, DBAASP_smiles_list),
                               total=len(DBAASP_smiles_list),
                               desc='FPs for DBAASP',
                               unit='mol'))
        gen_fps    = list(tqdm(p.imap(compute_fingerprint, generated_smiles_list),
                               total=len(generated_smiles_list),
                               desc='FPs for generated',
                               unit='mol'))

    # Ensure all SMILES strings were parsed correctly.
    if any(fp is None for fp in DBAASP_fps):
        raise ValueError("One or more SMILES strings in DBAASP could not be parsed.")

    if any(fp is None for fp in gen_fps):
        raise ValueError("One or more SMILES strings in generated mols could not be parsed.")

    n_gen = len(gen_fps)
    n_db = len(DBAASP_fps)
    # 只走上三角
    tasks = [(i, j, gen_fps[i], DBAASP_fps[j])
             for i in range(n_gen) for j in range(n_db)]

    # Preallocate an empty similarity matrix.
    sim_mat = np.zeros((n_gen, n_db), dtype=float)

    # 3. Compute similarity in parallel for each pair.
    print('\n Calculating similarity matrix ...')
    with Pool(processes=cpu_count()) as p:
        for i, j, sim in tqdm(p.imap(compute_similarity_pair, tasks, chunksize=256),
                              total=len(tasks),
                              desc='Comparing Similarity',
                              unit='pair'):
            sim_mat[i, j] = sim

    # Optionally, set the diagonal to 1.0 (self-similarity).
    # np.fill_diagonal(similarity_matrix, 1.0)

    return sim_mat

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
    # merged_smiles_path = current_dir / 'Data' / 'mol_visualize' / 'Colistin.txt'
    merged_smiles_path = current_dir / 'Data' / 'mol_visualize' / 'antibiotics.csv'
    DBAASP_ids, DBAASP_smiles_list = get_smiles_list(merged_smiles_path)
    # with open(merged_smiles_path, 'r') as f:
    #     DBAASP_smiles_list = [line.strip() for line in f]

    length_max_sim_list_dict = {}

    strain = '25922'
    MIC = 1
    length_list = [145]
    for length in length_list:
        generated_selfies_file = f'/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_SELFIES/strain_{strain}_MIC_{MIC}_length_{length}_noise.txt'
        with open(generated_selfies_file, 'r') as f:
            generated_selfies = f.read().splitlines()

        generated_smiles_list = []
        for selfies_str in generated_selfies:
            smiles_str = sf.decoder(selfies_str.strip())
            generated_smiles_list.append(smiles_str)


        sim_matrix = main(DBAASP_smiles_list, generated_smiles_list)

        length_max_sim_list_dict[str(length)] = np.max(sim_matrix, axis=1)
        # print("Similarity Matrix:")
        # print(f'max sims: {np.max(sim_matrix, axis=1)}')
        # print(f'min sims: {np.min(sim_matrix, axis=1)}')
        # print(f'mean sims: {np.mean(sim_matrix, axis=1)}')
        #
        # out_npy = current_dir / 'Data' / 'corrected_generated_mols' / "inhouse_DBAASP_and_gen_mol_sim_matrix.npy"
        #
        # np.save(out_npy, sim_matrix)
        # print('Done, saved to', out_npy)

    df_list = []
    for length, sims in length_max_sim_list_dict.items():
        # 确保 sims 是一维数组
        arr = np.asarray(sims).flatten()
        df = pd.DataFrame({
            'Length': [length] * len(arr),
            'MaxSim': arr
        })
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)

    colors = ["#FFFDD0", "#F7CFE1", "#B49EDE"]  # , "#759ECD"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors, N=len(length_list))

    fractions = np.linspace(0, 1, len(length_list))
    palette = [custom_cmap(f) for f in fractions]

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.violinplot(
        x='Length',
        y='MaxSim',
        data=df,
        inner='quartile',  # 显示四分位线
        scale='width',  # 每个 violin 的宽度相同
        cut=0,  # 不画超过数据范围的“胡须”
        palette=palette,
        width=0.5,
    )

    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=1.6)


    def lighten_color(color, amount=0.5):
        """
        将 color 提亮：color 可以是 RGB 或 RGBA tuple，也可以是 hex 字符串。
        amount 越大越接近白色（0–1 之间）。
        """
        # 把任何 color 转成 RGB tuple (r, g, b)
        c = mcolors.to_rgb(color)
        # 混合白色：(c + (1-c)*amount)
        return tuple(c_i + (1 - c_i) * amount for c_i in c)


    edge_colors = []
    for pc in [c for c in ax.collections if isinstance(c, PolyCollection)]:
        # 取出原填充色（RGBA），我们只要前 3 个 channels
        face_rgba = pc.get_facecolor()[0]
        face_rgb = face_rgba[:3]

        # amount 控制提亮程度，0.0 不变，1.0 全白
        edge_rgb = lighten_color(face_rgb, amount=-0.8)

        pc.set_edgecolor(edge_rgb)
        pc.set_linewidth(2.0)
        edge_colors.append(edge_rgb)

    for idx, line in enumerate(ax.lines):
        violin_idx = idx // 3  # 每 3 条线对应一个 violin
        line.set_color(edge_colors[violin_idx])
        line.set_linewidth(2.0)  # 可根据需要调粗细
        # line.set_linestyle('--')       # 如果想虚线也可以在这里再加

    ax.set_axisbelow(True)
    ax.set_ylim(0, 1)
    if MIC==1:
        ax.set_title(f"Max Similarity with DBAASP by\nGenerated Length targeting {strain}", fontsize=14)
    else:
        ax.set_title(f"Max Similarity with DBAASP by\nGenerated Length without Guidance", fontsize=14)
    x_label = ax.set_xlabel("Generated length")  # 去掉 x 轴标题
    x_label.set_fontsize(11)
    y_label = ax.set_ylabel("Max Similarity")  # 根据实际情况修改单位
    y_label.set_fontsize(11)

    plt.xticks(fontsize=10)

    sns.despine(fig=fig, ax=ax,
                top=True, right=True, bottom=True, left=True)

    ax.tick_params(axis='both', which='both', length=0)

    plt.tight_layout()
    # if MIC == 1:
    #     plt.savefig(f"/data2/tianang/projects/Synergy/paper_figs/{strain}-w-guide-mol-sim.pdf", format="pdf", bbox_inches="tight")
    # else:
    #     plt.savefig(f"/data2/tianang/projects/Synergy/paper_figs/{strain}-wo-guide-mol-sim.pdf", format="pdf", bbox_inches="tight")
    plt.show()