from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import time
import os


def select_param_value(name, default_value):
    return os.environ[name] if name in os.environ else default_value


def preload_embed(dir_location, bert_embed, read_train=True):
    if bert_embed:
        return preload_bert_embed(dir_location, read_train)
    else:
        return preload_wordvec_embed(dir_location)

def preload_wordvec_embed(dir_location):
    start = time.time()
    import dataloader
    embs =  dataloader.load_embedding(os.path.join(dir_location,"embedding_filtered"))
    print("took {} seconds".format(time.time()-start))
    print("preloaded embeddings from amazon dataset.")
    print("")
    return embs

def preload_bert_embed(data_dir_path, read_train=True):
    start = time.time()
    import dataloader
    data =  dataloader.read_bert(data_dir_path, read_train)
    print("took {} seconds".format(time.time()-start))
    print("preloaded bert embeddings.")
    print("")
    return data

def general_arg_parser():
    """ CLI args related to training and testing models. """
    p = ArgumentParser(add_help=False)
    p.add_argument("-d", '--base_data_dir', help="Data directory", type=str, required=True)
    p.add_argument("-a", "--dataset", help="Dataset name, including category if Amazon", type=str, required=True)
    p.add_argument("-p", "--pattern", help="Pattern specification", type=str, default="1-gram,2-gram,3-gram,4-gram")
    p.add_argument("--d_out", help="Output dimension(?)", type=str, default="0,4,0,2")
    p.add_argument("-g", "--gpu", help="Use GPU", action='store_true')
    p.add_argument('--depth', help="Depth of network", type=int, default=1)
    p.add_argument("-s", "--seed", help="Random seed", type=int, default=None)
    p.add_argument("-b", "--batch_size", help="Batch size", type=int, default=64)
    p.add_argument("--use_last_cs", help="Only use last hidden state as output value", action='store_true')
    p.add_argument("--weight_norm", help="Normalize weights", action='store_true')
    p.add_argument("--bert_embed", help="True if using BERT embeddings.", type=str, default="False")

    # p.add_argument("--max_doc_len",
    #                help="Maximum doc length. For longer documents, spans of length max_doc_len will be randomly "
    #                     "selected each iteration (-1 means no restriction)",
    #                type=int, default=-1)
    # p.add_argument("-n", "--num_train_instances", help="Number of training instances", type=int, default=None)
    # p.add_argument("-e", "--embedding_file", help="Word embedding file", required=True)

    return p

def str2bool(str):
  return str.lower() in ["true", "1"]

