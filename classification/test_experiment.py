import os
import sys
import experiment_tools
import train_classifier
from experiment_params import get_categories, ExperimentParams
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def main(argv):
    is_bert = experiment_tools.str2bool(experiment_tools.select_param_value('BERT_EMBED', argv.bert_embed))
    loaded_embedding = experiment_tools.preload_embed(os.path.join(argv.base_data_dir,argv.dataset), is_bert, False)

    models = argv.input_model.split(",")
    d_outs = argv.d_out.split("_")
    patterns = argv.pattern.split("_")

    for (model,d_out, pattern) in zip(models, d_outs, patterns):
        print("Checking model {} with pattern={} and d_out={}".format(model, d_out, pattern))
        # a basic experiment
        args = ExperimentParams(pattern = pattern, d_out = d_out,
                                    seed = argv.seed, loaded_embedding = loaded_embedding,
                                    dataset = argv.dataset, use_rho = False,
                                    depth = argv.depth, gpu=argv.gpu,
                                    batch_size=argv.batch_size, use_last_cs=argv.use_last_cs,
                                    base_data_dir = argv.base_data_dir, input_model=model,
                                    weight_norm = argv.weight_norm,
                                    bert_embed = is_bert)

        if argv.visualize > 0:
            train_classifier.main_visualize(args, os.path.join(argv.base_data_dir,argv.dataset), argv.visualize)
        else:
            _ = train_classifier.main_test(args)

    return 0
        


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            parents=[experiment_tools.general_arg_parser()])
    parser.add_argument("-m", "--input_model", help="Saved model file", required=True, type=str)
    parser.add_argument("-v", "--visualize", help="Visualize (rather than test): top_k phrases to visualize", type=int, default=0)
    sys.exit(main(parser.parse_args()))
