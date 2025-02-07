import numpy as np
import argparse
import pandas as pd

import sys
sys.path.insert(0,'/scratch/st-benbr-1/xijohnny/cd-ode-cocycle/loci')

from loci.causa.loci import loci

def standardize_data(x, return_statistics = False, trim_outliers = 0):
    """
    Standardize the data and do a box-trim of outliers (pure numpy version)).
    """
    if x.shape[1] > 2:
        x = x[:, 0:2]

    if trim_outliers > 0:
        x_ = x[:, 0]
        y_ = x[:, 1]
        x_low, x_high = np.quantile(x_, np.array([trim_outliers/2, 1 - trim_outliers/2]))
        y_low, y_high = np.quantile(y_, np.array([trim_outliers/2, 1 - trim_outliers/2]))
        x = x[(x_ > x_low) & (x_ < x_high) & (y_ > y_low) & (y_ < y_high)]


    mean = np.mean(x, axis = 0, keepdims = True)
    std = np.std(x, axis = 0, keepdims = True)
    if return_statistics:
        return (x - mean) / std, mean.squeeze(), std.squeeze()
    return x - mean / std

def parse_arguments():
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--dataset", metavar = "DATASET", type = str, default = "velocity", help = "Dataset to use for benchmarking, pick from ['tuebingen', 'sim', 'simc', 'simg', 'simln'].")
    parser.add_argument("--outlier_trim", metavar = "OUTLIER_TRIM", type = float, default = 0, help = "What percent of outliers to remove from data.")
    parser.add_argument("--n_steps", metavar = "N_STEPS", type = int, default = 1000, help = "Number of steps for optimization.")
    parser.add_argument("--model", metavar = "MODEL", type = str, default = "hsic", help = "Model to use for training (['nn', 'anm', lsnm', 'additive', 'parametric-lin', 'parametric-quad', 'parametric-cubic', 'parametric-quartic']).")
    parser.add_argument("--remote", action='store_true', default = False, help = "Running on remote server.")
    return parser.parse_args()

args = parse_arguments()


if args.remote:
    BASE_PATH = "/scratch/st-benbr-1/xijohnny/cd-ode-cocycle/data/"
else:
    BASE_PATH = "/Users/johnnyxi/Documents/phd/CausalDiscover/cd-cocycle-jax/data/"

if args.dataset == "tuebingen":
    DATA_PATH = BASE_PATH + "Tuebingen"
elif args.dataset == "sim":
    DATA_PATH = BASE_PATH + "Benchmark_simulated/SIM"
elif args.dataset == "simc":
    DATA_PATH = BASE_PATH + "Benchmark_simulated/SIM-c"
elif args.dataset == "simg":
    DATA_PATH = BASE_PATH + "Benchmark_simulated/SIM-G"
elif args.dataset == "simln":
    DATA_PATH = BASE_PATH + "Benchmark_simulated/SIM-ln"
elif args.dataset == "periodic":
    DATA_PATH = BASE_PATH + "new_sim/periodic"
elif args.dataset == "sigmoid":
    DATA_PATH = BASE_PATH + "new_sim/sigmoid"
elif args.dataset == "velocity":
    DATA_PATH = BASE_PATH + "new_sim/velocity_exp"
elif args.dataset == "an":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/AN"
elif args.dataset == "ans":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/AN-s"
elif args.dataset == "ls":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/LS"
elif args.dataset == "lss":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/LS-s"
elif args.dataset == "mnu":
    DATA_PATH = BASE_PATH + "ANLSMN_pairs/MN-U"

if __name__ == "__main__":

    success_weights = []
    total_weights = []
    scores = []

    if args.dataset in ["an", "ans", "ls", "lss", "mnu"]:
        meta = np.loadtxt(DATA_PATH + "/pairs_gt.txt")
    else:
        meta = pd.read_csv(DATA_PATH + "/pairmeta.txt",  delim_whitespace=True,
                            header=None,
                            names=['pair_id', 'cause_start', 'cause_end', 'effect_start', 'effect_end', 'weight'],
                            index_col=0)
        discrete_pairs = [47, 70, 107]
        multivariate_pairs = [52, 53, 54, 55, 71, 105]
        tue_blacklist = discrete_pairs + multivariate_pairs

    for iter in range(len(meta)):

        if args.dataset in ["an", "ans", "ls", "lss", "mnu"]:
            pair_id = iter + 1
            effect = meta[iter] + 1
            if effect == 1:
                cause = 2
            else:
                cause = 1
            weight = 1
            dat = np.loadtxt(DATA_PATH + f"/pair_{pair_id}.txt", delimiter = ",", skiprows = 1, usecols = (1,2))
        else: 
            pair_id = meta.index[iter].astype(int)
            if args.dataset=="tuebingen" and pair_id in tue_blacklist:
                continue
            cause = meta["cause_start"].values[iter].astype(int)
            effect = meta["effect_start"].values[iter].astype(int)
            weight = meta["weight"].values[iter]
            dat = np.loadtxt(DATA_PATH + f"/pair{pair_id:04d}.txt")

        print(f"Pair ID: {pair_id}")

        n = dat.shape[0]

        dat, dat_mean, dat_std = standardize_data(dat, return_statistics = True, trim_outliers = args.outlier_trim)
        x_data, x_data_mean, x_data_std = dat[:,0], dat_mean[0], dat_std[0]
        y_data, y_data_mean, y_data_std = dat[:,1], dat_mean[1], dat_std[1] 
        
        if args.model == "hsic":
            score = loci(np.array(x_data), np.array(y_data), n_steps = args.n_steps)
        elif args.model == "lik":
            score = loci(np.array(x_data), np.array(y_data), independence_test = False, n_steps = args.n_steps)
        elif args.model == "hsic_spline":
            score = loci(np.array(x_data), np.array(y_data), neural_network=False, n_steps = args.n_steps)
        elif args.model == "lik_spline":
            score = loci(np.array(x_data), np.array(y_data), independence_test = False, neural_network=False, n_steps = args.n_steps)

        scores.append(np.abs(score))
        if score > 0: 
            est_cause = 1
        else:
            est_cause = 2

        if est_cause == cause:
            success_weights.append(weight)
        else:
            success_weights.append(0)
        
        total_weights.append(weight)
        
        print(f"Running Success Rate: {sum(success_weights)/sum(total_weights)}, Pair ID: {pair_id}")

    ## AUDRC calculation

    idx = np.flip(np.argsort(scores))
    scores, success_weights, total_weights = np.array(scores)[idx], np.array(success_weights)[idx], np.array(total_weights)[idx]

    AUDRC = 0

    for i in range(len(scores)):
        AUDRC += sum(success_weights[:i+1]) / sum(total_weights[:i+1])
    
    AUDRC /= len(scores)

    print(f"AUDRC: {AUDRC}")
    print(f"Final Success Rate: {sum(success_weights)/sum(total_weights)}")