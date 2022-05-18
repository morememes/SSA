import numpy as np
import pandas as pd

import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.robjects.packages import importr

def predict_onepoint_by_ssa(seq: np.array, L: int, r: list, method: str, rssa: object):
    s = rssa.ssa(seq, kind='1d-ssa', L=L);
    res = rssa.rforecast(s, groups = robjects.r(f'list(1:{r})'), len = 1)
    #resD = dict(zip(res.names, list(res)))
    #pp = reduce(lambda x,y: x + y, list(resD['mean']))[0]"
    
    return res

def predict_array_by_ssa(seq: np.array, array: np.array, L: int, r: list, method: str, rssa: object) -> np.array:
    preds = []
    for value in array:
        preds.append(predict_onepoint_by_ssa(seq, L, r, method, rssa))
        seq = np.append(seq[1:], value)
    return np.array(preds)

def grid_search_ssa_params(train: np.array, val: np.array, L: list, r: list, methods: list, rssa: object) -> tuple:
    print(f'Всего {len(L) * len(r) * len(methods) } различных моделей.')
    rmse_error_best = 1000000000
    L_best = None
    r_best = None
    method_best = None
    results = []
    
    for method_curr in methods:
        for L_curr in L:
            for r_curr in r:
                if (L_curr >= r_curr):
                    preds = predict_array_by_ssa(train, val, L_curr, r_curr, method_curr, rssa)

                    rmse_error = np.sqrt(((preds.reshape(-1) - val)**2).mean())
                    results.append((rmse_error, L_curr, r_curr, method_curr))
                    
                    if (rmse_error < rmse_error_best):
                        L_best = L_curr
                        r_best = r_curr
                        method_best = method_curr
                        rmse_error_best = rmse_error
    
    return ((L_best, r_best, method_best), rmse_error_best, results)

def search_ssa_params(folder: str, path: str = 'data/ssa_res/',  **kwargs) -> tuple:
    results = grid_search_ssa_params(**kwargs)
    dfRes = pd.DataFrame(results[2], columns = ['rmse', 'L', 'r', 'method'])
    dfRes = dfRes.sort_values('rmse')
    
    if folder[-1] != '/':
        folder += '/'
    dfRes.to_csv(f'{path}{folder}results.csv', sep = ';', index=False)
    
    with open(f'{path}{folder}info.txt', 'w') as f:
        f.write(f"TRAIN_SIZE: {len(kwargs['train'])} \n")
        f.write(f"L list: {kwargs['L']} \n")
        f.write(f"r list: {kwargs['r']} \n \n")
        
        f.write(f"Best L: {results[0][0]} \n")
        f.write(f"Best r: {results[0][1]} \n")
        f.write(f"Best method: {results[0][2]} \n")
        f.write(f"Error: {results[1]} \n")
    return results
