import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from openbabel import pybel

column_name = 'H_vap'


def smiles_to_xyz(smiles):
    """
    将SMILES字符串转换为xyz格式的坐标
    """
    try:
        # 使用RDKit生成3D坐标
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 生成3D坐标
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # 获取原子数量
        num_atoms = mol.GetNumAtoms()
        
        # 构建xyz格式字符串
        xyz_lines = [str(num_atoms), ""]
        
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
        
        return "\n".join(xyz_lines)
    
    except Exception as e:
        print(f"Error converting SMILES {smiles}: {e}")
        return None

def convert_csv_to_molecules_json():
    """
    将CSV文件转换为molecules.json格式
    """
    # 读取CSV文件
    df = pd.read_csv(f'matched_compounds_data_{column_name}.csv')
    
    print(f"读取到 {len(df)} 条数据")
    
    # 准备数据字典
    molecules_data = {
        "xyz": {},
        column_name: {}  

    }
    
    successful_conversions = 0
    failed_conversions = 0
    
    for idx, row in df.iterrows():
        compound_name = row['compound_name']
        smiles = row['SMILES']
        boiling_point = row[column_name]
        
        # 转换SMILES到xyz
        xyz_coords = smiles_to_xyz(smiles)
        
        if xyz_coords is not None:
            # 使用索引作为键（类似原始molecules.json格式）
            key = str(idx)
            molecules_data["xyz"][key] = xyz_coords
            molecules_data[column_name][key] = boiling_point
            successful_conversions += 1
        else:
            failed_conversions += 1
            print(f"无法转换化合物: {compound_name} (SMILES: {smiles})")
        
        # 每处理100个化合物显示进度
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx + 1}/{len(df)} 个化合物")
    
    print(f"\n转换完成:")
    print(f"成功转换: {successful_conversions} 个化合物")
    print(f"转换失败: {failed_conversions} 个化合物")
    
    # 保存为JSON文件
    with open(f'molecules_{column_name}.json', 'w') as f:
        json.dump(molecules_data, f, indent=2)

    return molecules_data

if __name__ == "__main__":
    # 执行转换
    molecules_data = convert_csv_to_molecules_json()
    
