import random
import numpy as np
import pandas as pd
import untils.until as utils
from torch.utils.data import Dataset, DataLoader

SPLIT_CASE = 0
SEED = 42
BATCH_SIZE = 32
np.random.seed(SEED)
random.seed(SEED)

"""Load GDSC data."""
rma, var, GDSC_smiles = utils.load_GDSC_data()
GDSC_smiles_vals = GDSC_smiles["smiles"].values
GDSC_smiles_index = GDSC_smiles.index
GDSC_cell_names = rma.index.values
GDSC_gene = rma.columns.values

"""split dataset"""
data = pd.read_csv("../data/GDSC/GDSC_data/cell_drug_labels.csv", index_col=0)
train_id, test_id = utils.split_data(data, split_case=SPLIT_CASE, ratio=0.9, cell_names=GDSC_cell_names)

dataset_sizes = {'train': len(train_id), 'test': len(test_id)}
print(dataset_sizes['train'], dataset_sizes['test'])


class CreateDataset(Dataset):
    def __init__(self, rma_all, var_all, all_id):
        self.rma_all = rma_all
        self.var_all = var_all
        self.all_id = all_id.values

    def __len__(self):
        return len(self.all_id)

    def __getitem__(self, idx):
        cell_line_id = self.all_id[idx][0]
        drug_id = self.all_id[idx][1].astype('int')
        y = self.all_id[idx][2].astype('float32')
        rma = self.rma_all.loc[cell_line_id].values.astype('float32')
        var = self.var_all.loc[cell_line_id].values.astype('float32')
        return rma, var, drug_id, y


trainDataset = CreateDataset(rma, var, train_id)
train_loader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
for idx, (rma, var, drug_id, y) in enumerate(train_loader):
    print("Nothing")
    break

print("Finish")
