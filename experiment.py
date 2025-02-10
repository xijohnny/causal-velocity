import jax.numpy as jnp
import numpy as onp
import argparse
import pandas as pd

from training import fit_sm_bd
from utils import standardize_data, nullable_float, count_nparams
from models import parametric_model, nn_model, anm_model, lsnm_model, additive_model

def parse_arguments():
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("--dataset", metavar = "DATASET", type = str, default = "velocity", help = "Dataset to use for benchmarking. See data/ for options.")
    parser.add_argument("--seed", metavar = "SEED", type = int, default = 0, help = "Seed for parameter initialization.")
    parser.add_argument("--score", metavar = "SCORE", type = str, default = "stein", help = "Score estimation method (['stein', 'kde', 'hybrid]).")
    parser.add_argument("--kernel", metavar = "KERNEL", type = str, default = "gauss", help = "Kernel for score estimation (['gauss', 'exp']).")
    parser.add_argument("--model", metavar = "MODEL", type = str, default = "parametric-quad", help = "Model to use for training (['nn', 'anm', lsnm', 'additive', 'parametric-lin', 'parametric-quad', 'parametric-cubic', 'parametric-quartic']).")
    parser.add_argument("--add_fourier", action='store_true', default = False, help = "Add Fourier terms (only parametric models).")
    parser.add_argument("--add_exponential", action='store_true', default = False, help = "Add exponential terms (only parametric models).")
    parser.add_argument("--layers", metavar = "LAYERS", type = int, default = 2, help = "Number of layers for neural network models.")
    parser.add_argument("--hidden_size", metavar = "HIDDEN_SIZE", type = int, default = 32, help = "Hidden size for neural network models.")
    parser.add_argument("--init_weight", metavar = "INIT_WEIGHT", type = float, default = 0.2, help = "Initialization variance.")
    parser.add_argument("--reg", metavar = "REG", type = float, default = 0.1, help = "Regularization term for score estimator. Suggest 0.1 for stein and 0.01 for kde.")
    parser.add_argument("--optimizer", metavar = "OPTIMIZER", type = str, default = "adam", help = "Optimizer to use for training (['sgd', 'adam']).")
    parser.add_argument("--lr", metavar = "LR", type = float, default = 0.1, help = "Learning rate for optimizer.")
    parser.add_argument("--n_steps", metavar = "N_STEPS", type = int, default = 100, help = "Number of steps for optimization.")
    parser.add_argument("--loss_l2", metavar = "LOSS_L2", type = float, default = 0.00001, help = "L2 regularization term for loss function.")
    parser.add_argument("--lam_complexity", metavar = "LAM_COMPLEXITY", type = float, default = 0.01, help = "Complexity penalty derivative.")
    parser.add_argument("--complexity_order", metavar = "COMPLEXITY_ORDER", type = int, default = 1, help = "Order of complexity penalty. Only order 1 is currently implemented.")
    parser.add_argument("--loss_pos", metavar = "LOSS_POS", type = str, default = "squared", help = "Positivity transformation for loss function (['abs', 'squared']).")
    parser.add_argument("--val_split", metavar = "VAL_SPLIT", type = nullable_float, help = "Validation split (for model selection).")
    parser.add_argument("--test_split", metavar = "TEST_SPLIT", type = nullable_float, help = "Test split (to determine direction).")
    parser.add_argument("--gof_eval", metavar = "GOF_EVAL", type = str, default = "raw", help = "GoF evaluation metric, either use raw or sq GoF (['raw', 'sq']).")
    parser.add_argument("--outlier_trim", metavar = "OUTLIER_TRIM", type = float, default = 0, help = "What percent of outliers to remove from data.")
    parser.add_argument("--n_reinit", metavar = "N_REINIT", type = int, default = 1, help = "Number of reinitializations for optimization.")
    return parser.parse_args()

args = parse_arguments()
SEED = args.seed
LAYERS = args.layers
HIDDEN_SIZE = args.hidden_size
INIT_WEIGHT = args.init_weight
REG = args.reg
LR = args.lr
VAL_SPLIT = args.val_split
TEST_SPLIT = args.test_split
OUTLIER_TRIM = args.outlier_trim
LOSS_L2 = args.loss_l2
N_STEPS = args.n_steps
SCORE = args.score
LOSS_POS = args.loss_pos
KERNEL = args.kernel
GOF_EVAL = args.gof_eval
OPTIMIZER = args.optimizer
MODEL = args.model
N_REINIT = args.n_reinit
LAM_COMPLEXITY = args.lam_complexity
COMPLEXITY_ORDER = args.complexity_order

BASE_PATH = "data/"

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


if LAYERS == 1:
    ## linear. 
    HIDDEN_SIZE = 1


if __name__ == "__main__":

    if MODEL=="parametric-lin":
        model = parametric_model(basis_name = "linear")
    elif MODEL=="parametric-quad":
        model = parametric_model(basis_name = "quadratic")
    elif MODEL=="parametric-cubic":
        model = parametric_model(basis_name = "cubic")
    elif MODEL=="parametric-quartic":
        model = parametric_model(basis_name = "quartic")
    elif MODEL=="nn":
        model = nn_model(layers = LAYERS, hidden_size = HIDDEN_SIZE)
    elif MODEL=="anm":
        model = anm_model(layers = LAYERS, hidden_size = HIDDEN_SIZE)
    elif MODEL=="lsnm":
        model = lsnm_model(layers = LAYERS, hidden_size = HIDDEN_SIZE)
    elif MODEL=="additive":
        model = additive_model(layers = LAYERS, hidden_size = HIDDEN_SIZE)

    if MODEL in ["parametric-lin", "parametric-quad", "parametric-cubic", "parametric-quartic"]:
        if args.add_exponential:
            model.add_exponential_terms = True
            model.nparams += 3
        if args.add_fourier:
            model.add_fourier_terms = True
            model.nparams += 4

    success_weights = []
    total_weights = []
    scores = []

    if args.dataset in ["an", "ans", "ls", "lss", "mnu"]:
        meta = onp.loadtxt(DATA_PATH + "/pairs_gt.txt")
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
            dat = onp.loadtxt(DATA_PATH + f"/pair_{pair_id}.txt", delimiter = ",", skiprows = 1, usecols = (1,2))
        else: 
            pair_id = meta.index[iter].astype(int)
            if args.dataset=="tuebingen" and pair_id in tue_blacklist:
                continue
            cause = meta["cause_start"].values[iter].astype(int)
            effect = meta["effect_start"].values[iter].astype(int)
            weight = meta["weight"].values[iter]
            dat = onp.loadtxt(DATA_PATH + f"/pair{pair_id:04d}.txt")

        n = dat.shape[0]

        dat, dat_mean, dat_std = standardize_data(dat, return_statistics = True, trim_outliers = OUTLIER_TRIM)
        x_data, x_data_mean, x_data_std = dat[:,0], dat_mean[0], dat_std[0]
        y_data, y_data_mean, y_data_std = dat[:,1], dat_mean[1], dat_std[1] 
        
        best_gof_fwd = 99999999
        best_gof_bwd = 99999999

        for i in range(N_REINIT):

            def V_MODEL(y, t, params):
                return model(y, t, params)
        
            PARAMS_INIT = model.params_init(seed = SEED + iter*N_REINIT + i, init_weight = INIT_WEIGHT)

            gof_fwd, gof_bwd, complexity_fwd, complexity_bwd, params_fwd, params_bwd = fit_sm_bd(PARAMS_INIT, V_MODEL, x_data, y_data, 
                                            lr = LR / jnp.log(count_nparams(PARAMS_INIT)), 
                                            n_steps = N_STEPS,
                                            reg = REG, 
                                            score = SCORE,
                                            gof = GOF_EVAL, 
                                            score_kernel = KERNEL, 
                                            loss_pos = LOSS_POS, 
                                            val_split=VAL_SPLIT,
                                            test_split=TEST_SPLIT, 
                                            loss_l2 = LOSS_L2,
                                            lam_complexity=LAM_COMPLEXITY,
                                            optimizer = OPTIMIZER,
                                            complexity_order = COMPLEXITY_ORDER)

            if gof_fwd < best_gof_fwd:
                best_gof_fwd = gof_fwd
                best_params_fwd = params_fwd
                best_complexity_fwd = complexity_fwd
            if gof_bwd < best_gof_bwd:
                best_gof_bwd = gof_bwd
                best_params_bwd = params_bwd
                best_complexity_bwd = complexity_bwd

        score = (best_gof_bwd) - (best_gof_fwd)
        scores.append(onp.abs(score))
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

    idx = onp.flip(onp.argsort(scores))
    scores, success_weights, total_weights = onp.array(scores)[idx], onp.array(success_weights)[idx], onp.array(total_weights)[idx]

    AUDRC = 0

    for i in range(len(scores)):
        AUDRC += sum(success_weights[:i+1]) / sum(total_weights[:i+1])
    
    AUDRC /= len(scores)

    print(f"AUDRC: {AUDRC}")
    print(f"Final Success Rate: {sum(success_weights)/sum(total_weights)}")