import pickle
import random
import numpy as np
import pandas as pd
from untils import until as utils
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor

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
train_id, test_id = utils.split_data(data, split_case=SPLIT_CASE, ratio=0.95, cell_names=GDSC_cell_names)

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


def create_array_dataset(rma_all, smiles_latent_all, all_id):
    smiles_latent_id = [a['id'] for a in smiles_latent_all]

    features = []
    Y = []
    all_id = all_id.values

    for i in tqdm(range(len(all_id))):
        idx = all_id[i]
        cell_line_id = idx[0]
        rma = rma_all.loc[cell_line_id].values.astype('float32')
        y = idx[2]
        drug_id = idx[1]
        smiles_latent = smiles_latent_all[smiles_latent_id.index(drug_id)]['vec'][0]

        combined = np.concatenate((rma, smiles_latent), axis=0)

        features.append(combined)
        Y.append(y)

    return features, Y


# trainDataset = CreateDataset(rma, var, train_id)
# train_loader = DataLoader(dataset=trainDataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
# for idx, (rma, var, drug_id, y) in enumerate(train_loader):
#     print("Nothing")
#     break


with open("../data/GDSC/GDSC_data/GDSC_smiles_latent_all.pkl", "rb") as p_f:
    smiles_latent = pickle.load(p_f)

# X, Y = create_array_dataset(rma_all=rma, smiles_latent_all=smiles_latent, all_id=test_id)
# with open('../data/GDSC/GDSC_data/GDS_combined_test.pkl', "wb") as p_f:
#     pickle.dump([X, Y], p_f)
#
#
# X, Y = create_array_dataset(rma_all=rma, smiles_latent_all=smiles_latent, all_id=train_id)
# with open('../data/GDSC/GDSC_data/GDS_combined_train.pkl', "wb") as p_f:
#     pickle.dump([X, Y], p_f)

X, Y = create_array_dataset(rma_all=rma, smiles_latent_all=smiles_latent, all_id=train_id)

model = XGBRegressor()
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=cv, n_jobs=2)
scores = np.absolute(scores)
print('Mean MSE: %.3f (%.3f)' % (scores.mean(), scores.std()))
print(scores)