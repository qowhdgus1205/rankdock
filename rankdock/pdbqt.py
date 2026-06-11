import os
import argparse
import subprocess
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_pdbqt(smi_idx):
    smi, idx, output_dir = smi_idx
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # 수소 추가 및 3D 좌표 생성
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)

        # PDB 파일 임시 저장
        pdb_path = os.path.join(output_dir, f"molecule_{idx}.pdb")
        with open(pdb_path, "w") as f:
            f.write(Chem.MolToPDBBlock(mol))

        # PDB → PDBQT 변환 (Open Babel 사용)
        pdbqt_path = pdb_path.replace(".pdb", ".pdbqt")
        subprocess.run([
            "obabel", "-ipdb", pdb_path, "-opdbqt",
            "-O", pdbqt_path,
            "--gen3D",
            "--partialcharge", "gasteiger",
            "--deleteh"
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)

        os.remove(pdb_path)  # 임시 PDB 삭제
        return pdbqt_path
    except Exception:
        return None

def main(args):
    import pandas as pd
    smiles_df = pd.read_csv(args.input_csv)
    smiles_list = smiles_df[args.smiles_column].tolist()

    os.makedirs(args.output_dir, exist_ok=True)

    task_list = [(smi, i, args.output_dir) for i, smi in enumerate(smiles_list)]

    with Pool(processes=args.cores) as pool:
        list(tqdm(pool.imap_unordered(smiles_to_pdbqt, task_list), total=len(task_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SMILES to PDBQT in parallel")
    parser.add_argument("--input_csv", required=True, help="Input CSV file with SMILES column")
    parser.add_argument("--smiles_column", default="SMILES", help="Name of SMILES column")
    parser.add_argument("--output_dir", required=True, help="Directory to save .pdbqt files")
    parser.add_argument("--cores", type=int, default=cpu_count(), help="Number of parallel processes")
    args = parser.parse_args()

    main(args)