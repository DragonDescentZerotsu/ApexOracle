from rdkit.Chem import Descriptors


descriptor_names = [name for name, _ in Descriptors.descList if name != "Ipc"]

print(descriptor_names)