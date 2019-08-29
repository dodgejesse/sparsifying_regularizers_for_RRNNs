from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import sys
import os
from experiment_params import ExperimentParams
import train_classifier
import experiment_tools
import regularization_search_experiments

def main(argv):
    loaded_embedding = experiment_tools.preload_embed(os.path.join(argv.base_dir,argv.dataset))
    
    if argv.random_selection or 'RANDOM_SELECTION' in os.environ:
        hyper_parameters_assignments = regularization_search_experiments.hparam_sample()
    else:
        hyper_parameters_assignments = {
            "clip_grad": argv.clip,
            "dropout": argv.dropout,
            "rnn_dropout": argv.rnn_dropout,
            "embed_dropout": argv.embed_dropout,
            "lr": argv.lr, "weight_decay": argv.weight_decay,
            "depth": argv.depth
        }

    parameters = {
        'pattern': experiment_tools.select_param_value('PATTERN', argv.pattern),
        'd_out': experiment_tools.select_param_value('D_OUT', argv.d_out),
        'seed': int(experiment_tools.select_param_value('SEED', argv.seed)),
        'learned_structure': experiment_tools.select_param_value('LEARNED_STRUCTURE', argv.learned_structure),
        'semiring': experiment_tools.select_param_value('SEMIRING', argv.semiring)
    }

    kwargs = {
        "reg_goal_params": argv.reg_goal_params,
        "filename_prefix": argv.filename_prefix,
        "loaded_embedding": loaded_embedding,
        "dataset": argv.dataset, "use_rho": False,
        "gpu": argv.gpu,
        "max_epoch": argv.max_epoch, "patience": argv.patience,
        "batch_size": argv.batch_size, "use_last_cs": argv.use_last_cs,
        "logging_dir": argv.logging_dir,
        "base_data_dir": argv.base_dir, "output_dir": argv.model_save_dir,
        "reg_strength": argv.reg_strength, "sparsity_type": argv.sparsity_type
    }

    args = ExperimentParams(**kwargs, **parameters, **hyper_parameters_assignments)

    print(args)

    _ = train_classifier.main(args)


def training_arg_parser():
    """ CLI args related to training models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("--learned_structure", help="Learned structure", type=str, default="l1-states-learned")
    p.add_argument('--reg_goal_params', type=int, default = 20)
    p.add_argument('--filename_prefix', help='logging file prefix?', type=str, default="all_cs_and_equal_rho/saving_model_for_interpretability/")
    p.add_argument("-t", "--dropout", help="Use dropout", type=float, default=0.1943)
    p.add_argument("--rnn_dropout", help="Use RNN dropout", type=float, default=0.0805)
    p.add_argument("--embed_dropout", help="Use RNN dropout", type=float, default=0.3489)
    p.add_argument("-l", "--lr", help="Learning rate", type=float, default=2.553E-02)
    p.add_argument("--clip", help="Gradient clipping", type=float, default=1.09)
    p.add_argument('-w', "--weight_decay", help="Weight decay", type=float, default=1.64E-06)
    p.add_argument("-m", "--model_save_dir", help="where to save the trained model", type=str)
    p.add_argument("--logging_dir", help="Logging directory", type=str, required=True)
    p.add_argument("--max_epoch", help="Number of iterations", type=int, default=500)
    p.add_argument("--patience", help="Patience parameter (for early stopping)", type=int, default=30)
    p.add_argument("--sparsity_type", help="Type of sparsity (wfsa, edges, states, rho_entropy or none)",
                   type=str, default="none")
    p.add_argument("--reg_strength", help="Regularization strength", type=float, default=0.0)
    p.add_argument("--semiring", help="Type of semiring (plus_times, max_times, max_plus)",
                   type=str, default="plus_times")
    p.add_argument("--random_selection", help="Randomly select hyperparameters", action='store_true')
    # p.add_argument("-r", "--scheduler", help="Use reduce learning rate on plateau schedule", action='store_true')
    # p.add_argument("--debug", help="Debug", type=int, default=0)
    return p


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[experiment_tools.general_arg_parser(), training_arg_parser()])
    sys.exit(main(parser.parse_args()))
