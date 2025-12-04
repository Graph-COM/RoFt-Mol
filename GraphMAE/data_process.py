import torch
from torch_geometric.data import Data

def bump(g):
    return Data.from_dict(g.__dict__)

def main():
    old_data = torch.load("./datasets/molecule_net/toxcast/processed/geometric_data_processed.pt")
    new_data = (bump(old_data[0]), old_data[1])
    torch.save(new_data, "./datasets/molecule_net/toxcast/processed/geometric_data_processed_new.pt")
    print("g")

if __name__ == "__main__":
    main()