#!/usr/bin/env python

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys
import os
import copy
import time
import experiment_tools
import regularization_search_experiments
from experiment_params import ExperimentParams


def main(argv):
    is_bert = experiment_tools.str2bool(experiment_tools.select_param_value('BERT_EMBED', argv.bert_embed))

    loaded_embedding = experiment_tools.preload_embed(os.path.join(argv.base_data_dir, argv.dataset), is_bert, True)

    seed = experiment_tools.select_param_value('SEED', argv.seed)
    if seed is not None:
        seed = int(seed)
    training_args = {
        'pattern': experiment_tools.select_param_value('PATTERN', argv.pattern),
        'd_out': experiment_tools.select_param_value('D_OUT', argv.d_out),
        'seed': seed,
        'semiring': experiment_tools.select_param_value('SEMIRING', argv.semiring),
        "depth": argv.depth,
        "filename_prefix": argv.filename_prefix,
        "dataset": experiment_tools.select_param_value('DATASET', argv.dataset),
        "use_rho": False,
        "gpu": argv.gpu,
        "max_epoch": argv.max_epoch, "patience": argv.patience,
        "batch_size": int(experiment_tools.select_param_value('BATCH_SIZE', argv.batch_size)),
        "use_last_cs": argv.use_last_cs,
        "logging_dir": argv.logging_dir,
        "reg_strength": float(experiment_tools.select_param_value('REG_STRENGTH', argv.reg_strength)),
        "base_data_dir": argv.base_data_dir,
        "bert_embed": is_bert
    }


    # if reg_goal_params is not False, it should a comma-separated string.
    reg_goal_params = experiment_tools.select_param_value('REG_GOAL_PARAMS', argv.reg_goal_params)
    if reg_goal_params != "False":
        reg_goal_params = [int(x) for x in reg_goal_params.split(",")]
    else:
        reg_goal_params = False
        
    rand_search_args = {
        "k": argv.k,
        "l": argv.l,
        "m": argv.m,
        "n": argv.n,
        "sparsity_type": experiment_tools.select_param_value('SPARSITY_TYPE', argv.sparsity_type),
        "reg_goal_params_list": reg_goal_params,
        "distance_from_target": int(experiment_tools.select_param_value('DISTANCE_FROM_TARGET', argv.distance_from_target))
    }

    print(training_args)
    print(rand_search_args)

    training_args["loaded_embedding"] = loaded_embedding

    if argv.baseline:
        run_baseline_experiment(training_args, rand_search_args)
    else:
        run_random_search(training_args, rand_search_args)

def run_baseline_experiment(training_args, rand_search_args):
    start_time = time.time()
    counter = [0]

    total_evals = rand_search_args["m"] + rand_search_args["n"]
    
    assert training_args["reg_strength"] == 0, "No regularization for baseline experiments"
    assert rand_search_args["sparsity_type"] == "none", "No regularization for baseline experiments"
    assert not rand_search_args["reg_goal_params_list"], "No regularization for baseline experiments"

    args = regularization_search_experiments.train_m_then_n_models(m=rand_search_args["m"],
                                                                   n=rand_search_args["n"], counter=counter,
                                                                   total_evals=total_evals, start_time=start_time,
                                                                   **training_args)

def run_random_search(training_args, rand_search_args):
    start_time = time.time()
    counter = [0]

    total_evals = (rand_search_args["m"] + rand_search_args["n"] + \
                   rand_search_args["k"] + rand_search_args["l"]) * \
                  len(rand_search_args["reg_goal_params_list"])

    all_reg_search_counters = []

    for reg_goal_params in rand_search_args["reg_goal_params_list"]:
        best, reg_search_counters = regularization_search_experiments.train_k_then_l_models(
            k=rand_search_args["k"], l=rand_search_args["l"],
            counter=counter, total_evals=total_evals, start_time=start_time,
            reg_goal_params=reg_goal_params,
            sparsity_type=rand_search_args["sparsity_type"],
            distance_from_target=rand_search_args["distance_from_target"],
            **training_args)

        all_reg_search_counters.append(reg_search_counters)
        
        training_args_copy = get_unregularized_args(training_args, best)
        
        args = regularization_search_experiments.train_m_then_n_models(
            m=rand_search_args["m"], n=rand_search_args["n"], counter=counter,
            total_evals=total_evals, start_time=start_time,
            **training_args_copy)

    print("search counters:")
    for search_counter in all_reg_search_counters:
        print(search_counter)

# to get the args for an unregularized experiment
def get_unregularized_args(training_args, best):
    loaded_emb = training_args["loaded_embedding"]
    training_args["loaded_embedding"] = None
    training_args_copy = copy.deepcopy(training_args)
    training_args["loaded_embedding"] = loaded_emb
    training_args_copy["loaded_embedding"] = loaded_emb

    training_args_copy["pattern"] = best['learned_pattern']
    training_args_copy["d_out"] = best['learned_d_out']
    training_args_copy["learned_structure"] = 'l1-states-learned'
    del training_args_copy["reg_strength"]

    return training_args_copy

def training_arg_parser():
    """ CLI args related to training models. """
    p = ArgumentParser(add_help=False)
    p.add_argument('--reg_goal_params', default="80,60,40,20")
    p.add_argument('--filename_prefix', help='logging file prefix', type=str,
                   default="all_cs_and_equal_rho/saving_model_for_interpretability/")
    p.add_argument("--logging_dir", help="Logging directory", type=str, required=True)
    p.add_argument("--max_epoch", help="Max number of epochs", type=int, default=500)
    p.add_argument("--patience", help="Patience parameter (for early stopping)", type=int, default=30)
    p.add_argument("--sparsity_type", help="Type of sparsity (wfsa, edges, states, rho_entropy or none)",
                   type=str, default="states")
    p.add_argument("--reg_strength", help="Regularization strength", type=float, default=8 * 10 ** -6)
    p.add_argument("--semiring", help="Type of semiring (plus_times, max_times, max_plus)",
                   type=str, default="plus_times")
    p.add_argument("--k", help="K argument for random search", type=int, default=20)
    p.add_argument("--l", help="L argument for random search", type=int, default=5)
    p.add_argument("--m", help="M argument for random search", type=int, default=20)
    p.add_argument("--n", help="N argument for random search", type=int, default=5)
    p.add_argument("--distance_from_target", help="Distance from target goal allowed in random search", type=int, default=10)
    p.add_argument("--baseline", help="For running baselines.", action="store_true")
    
    return p


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[experiment_tools.general_arg_parser(), training_arg_parser()])
    sys.exit(main(parser.parse_args()))

