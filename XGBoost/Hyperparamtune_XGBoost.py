import pickle
import random
import numpy as np
import pandas as pd
from untils import until as utils
from tqdm import tqdm
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from typing import List, Dict

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
train_id, test_id = utils.split_data(data, split_case=SPLIT_CASE, ratio=0.8, cell_names=GDSC_cell_names)

dataset_sizes = {'train': len(train_id), 'test': len(test_id)}
print(dataset_sizes['train'], dataset_sizes['test'])


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


with open("../data/GDSC/GDSC_data/GDSC_smiles_latent_all.pkl", "rb") as p_f:
    smiles_latent = pickle.load(p_f)

X_train, Y_train = create_array_dataset(rma_all=rma, smiles_latent_all=smiles_latent, all_id=train_id)
X_valid, Y_valid = create_array_dataset(rma_all=rma, smiles_latent_all=smiles_latent, all_id=test_id)

space = {
    'max_depth': hp.quniform("max_depth", 3, 18, 1),
    'gamma': hp.uniform('gamma', 1, 9),
    'reg_alpha': hp.quniform('reg_alpha', 40, 180, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'n_estimators': hp.quniform('n_estimators', 1000, 1800, 10),
    'eta': hp.uniform('eta', 0.01, 0.3),
    'seed': 0

}


def objective(space: Dict[str, any], X_train=X_train, Y_train=Y_train,
              X_valid=X_valid, Y_valid=Y_valid) -> Dict[str, any]:
    clf = XGBRegressor(n_estimators=int(space['n_estimators']), max_depth=int(space['max_depth']), eta=space['eta'],
                       subsample=0.7, colsample_bytree=int(space['colsample_bytree']), objective='reg:squarederror',
                       tree_method='gpu_hist', gpu_id=0, predictor='cpu_predictor', random_state=42,
                       gamma=space['gamma'], reg_alpha=int(space['reg_alpha']), reg_lambda=space['reg_lambda'],
                       min_child_distance=int(space['min_child_weight']))

    evaluation = [(X_train, Y_train), (X_valid, Y_valid)]

    clf.fit(X_train, Y_train, eval_set=evaluation, eval_metric=mean_squared_error, early_stopping_rounds=10,
            verbose=False)

    preds = clf.predict(X_valid)
    mse = mean_squared_error(Y_valid, preds)
    print("[VALIDATION] MSE = ", mse)
    return {'loss': mse, 'status': STATUS_OK}


# model = XGBRegressor(n_estimators=1200, max_depth=10, eta=0.02, subsample=0.7, colsample_bytree=0.45,
#                      objective='reg:squarederror', tree_method='gpu_hist', gpu_id=0, predictor='cpu_predictor',
#                      random_state=42, gamma=1, reg_alpha=1.5, reg_lambda=5)


trials = Trials()
best_hyperparams = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)
print('Best Parameters found = ', best_hyperparams)


print("---------------------------------------->>>>>>>>>>>>>> Trials")
print(trials)


with open('trials.pkl', 'wb') as pkl_f:
    pickle.dump(trials, pkl_f)
