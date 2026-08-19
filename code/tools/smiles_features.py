import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator

# Initialize the generator once globally to prevent RDKit deprecation warnings
fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

def smiles_to_feature_vector(smiles):
    # Default zero vector if SMILES is missing/invalid
    default_vec = np.zeros(1025, dtype=float)
    
    if pd.isna(smiles) or not isinstance(smiles, str):
        return default_vec
        
    # try:
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is not None:
    #         # Generate 1024-bit Morgan Fingerprint (Radius 2)
    #         fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    #         fp_array = np.array(fp, dtype=float)
    #         # Recreate your [1.0] + fingerprint structure
    #         return np.concatenate(([1.0], fp_array))
    # except Exception:
    #     pass

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp_array = np.array(fp_gen.GetFingerprint(mol))
            # Recreate your [1.0] + fingerprint structure
            return np.concatenate(([1.0], fp_array))
    except Exception:
        pass
    
    return default_vec