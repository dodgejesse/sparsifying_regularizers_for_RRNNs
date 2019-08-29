import argparse
import os
import time
import math
import sys
import numpy as np
import torch.nn as nn
from torch.optim import SGD
from torch.autograd import Variable

if ".." not in sys.path:
    sys.path.append("..")
if "." not in sys.path:
    sys.path.append(".")
import rrnn
from semiring import *

if "../classification/" not in sys.path:
    sys.path.append("../classification/")

if "classification/" not in sys.path:
    sys.path.append("classification/")
import save_learned_structure

import train_classifier


SOS, EOS = "<s>", "</s>"

def read_corpus(path, sos=None, eos="</s>", shuffle=False):
    data = [ ]
    if sos is not None:
        data = [sos]
    with open(path) as fin:
        lines = [line.split() + [ eos ] for line in fin]
    if shuffle:
        np.random.shuffle(lines)
    for line in lines:
        data += line
    return data


def create_batches(data_text, map_to_ids, batch_size, cuda=False):
    data_ids = map_to_ids(data_text)
    N = len(data_ids)
    L = ((N-1) // batch_size) * batch_size
    x = np.copy(data_ids[:L].reshape(batch_size,-1).T)
    y = np.copy(data_ids[1:L+1].reshape(batch_size,-1).T)
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    x, y = x.contiguous(), y.contiguous()
    if cuda:
        x, y = x.cuda(), y.cuda()
    return x, y


class EmbeddingLayer(nn.Module):
    def __init__(self, emb_size, words, sos=SOS, fix_emb=False):
        super(EmbeddingLayer, self).__init__()
        word2id, id2word = {}, {}
        if sos not in word2id:
            word2id[sos] = len(word2id)
            id2word[word2id[sos]] = sos
        for w in words:
            if w not in word2id:
                word2id[w] = len(word2id)
                id2word[word2id[w]] =w
        self.word2id, self.id2word = word2id, id2word
        self.n_V, self.emb_size = len(word2id), emb_size
        self.embedding = nn.Embedding(self.n_V, emb_size)
        self.sosid = word2id[sos]

    def forward(self, x):
        return self.embedding(x)

    def map_to_ids(self, text):
        return np.asarray([self.word2id[x] for x in text],
                 dtype="int64")

    def map_to_tokens(self, ids):
        return [self.id2word[x] for x in ids.cpu().numpy()]


class Model(nn.Module):
    def __init__(self, words, args):
        super(Model, self).__init__()
        self.args = args

        self.emb_size = args.emb_size
        self.depth = args.depth
        self.drop = nn.Dropout(args.dropout)
        self.input_drop = nn.Dropout(args.input_dropout)
        self.output_drop = nn.Dropout(args.output_dropout)
        self.emb_layer = EmbeddingLayer(self.emb_size, words)
        self.n_V = self.emb_layer.n_V
        self.num_mlp_layer = args.num_mlp_layer

        use_tanh, use_relu, use_selu = 0, 0, 0
        if args.activation == "tanh":
            use_tanh = 1
        elif args.activation == "relu":
            use_relu = 1
        elif args.activation == "selu":
            use_selu = 1
        else:
            assert args.activation == "none"

        if args.model == "lstm":
            self.encoder=nn.LSTM(
                self.emb_size, self.emb_size,
                self.depth,
                dropout = args.rnn_dropout
            )
        elif args.model == "rrnn":
            first_layer_d_in = args.d_out.split(";")[0]
            final_layer_d_out = args.d_out.split(";")[-1]
            # num_wfsa_input = sum([int(one_size) for one_size in first_layer_d_in.split(",")])
            num_wfsa_output = sum([int(one_size) for one_size in final_layer_d_out.split(",")])

            # self.input_layer = nn.Linear(self.emb_size, num_wfsa_input)
            self.output_layer = nn.Linear(self.emb_size, self.n_V)
            if args.semiring == "plus_times":
                self.semiring = PlusTimesSemiring
            elif args.semiring == "max_plus":
                self.semiring = MaxPlusSemiring
            else:
                assert False, "Semiring should either be plus_times or max_plus, not {}".format(args.semiring)
            self.encoder = rrnn.RRNN(
                self.semiring,
                self.emb_size,
                args.d_out,
                self.depth,
                pattern=args.pattern,
                dropout=args.dropout,
                rnn_dropout=args.rnn_dropout,
                use_tanh=use_tanh,
                use_relu=use_relu,
                use_selu=use_selu,
                layer_norm=args.use_layer_norm,
                use_output_gate=args.use_output_gate
            )
        else:
            assert False

        if args.num_mlp_layer == 2:
            self.hidden = nn.Linear(num_wfsa_output, self.emb_size)
        elif args.num_mlp_layer == 1:
            assert False, "we want to use 2-layer mlps"
            pass
        else:
            assert False

        assert self.output_layer.weight.shape == self.emb_layer.embedding.weight.shape, "suspected this would be a problem."
        # tie weights
        self.output_layer.weight = self.emb_layer.embedding.weight

        self.init_weights()
        if args.model != "lstm":
           self.encoder.init_weights()


    def init_input(self, batch_size):
        args = self.args
        sosid = self.emb_layer.sosid
        init_input = torch.from_numpy(np.array(
            [sosid, sosid, sosid, sosid] * batch_size, dtype=np.int64).reshape(4, batch_size))

        if args.gpu:
            init_input = init_input.cuda()
        return Variable(init_input)


    def init_weights(self):
        val_range = (3.0/self.emb_size)**0.5
        for p in self.parameters():
            if p.dim() > 1:  # matrix
                p.data.uniform_(-val_range, val_range)
            else:
                p.data.zero_()


    def forward(self, x, init):
        emb = self.input_drop(self.emb_layer(x))
        # emb = self.drop(self.input_layer(emb).tanh())
        output, hidden, _ = self.encoder(emb, init)

        if self.num_mlp_layer == 2:
            output = self.drop(output)
            output = output.view(-1, output.size(2))
            output = self.hidden(output).tanh()
            output = self.output_drop(output)
        elif self.num_mlp_layer == 1:
            output = self.output_drop(output)
            output = output.view(-1, output.size(2))
        output = self.output_layer(output)
        return output, hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        if self.args.model == "lstm":
            return (Variable(weight.new(self.depth, batch_size, self.emb_size).zero_()),
                    Variable(weight.new(self.depth, batch_size, self.emb_size).zero_()))
        elif self.args.model == "rrnn":
            init_input = self.init_input(batch_size)
            emb = self.input_drop(self.emb_layer(init_input))
            # emb = self.drop(self.input_layer(emb).tanh())
            output, hidden, _ = self.encoder(emb, None)
            return hidden
        else:
            assert False


    def compute_loss(self, emb, y):
        batch_size = 1
        hidden = self.init_hidden(batch_size)
        hidden = repackage_hidden(self.args, hidden)
        output, hidden, _ = self.encoder(emb, hidden)
        output = output.view(-1, output.size(2))
        output = self.output_layer(output)

        criterion = nn.CrossEntropyLoss(size_average=False)
        loss = criterion(output, y) / emb.size(1)
        print (loss)
        loss.backward()
        return loss


    def print_pnorm(self):
        norms = [ "{:.0f}".format(x.norm().data[0]) for x in self.parameters() ]
        print_and_log("\tp_norm: {}\n".format(norms), self.logging_file)


def repackage_hidden(args, hidden):
    if args.model == "lstm":
        return (Variable(hidden[0].data), Variable(hidden[1].data))
    elif args.model == "rrnn":
        if args.pattern == "bigram":
            return (Variable(hidden[0].data), Variable(hidden[1].data))
        elif args.pattern == "unigram":
            return Variable(hidden.data)
        else:
            for i, hss in enumerate(hidden):
                for j, hs in enumerate(hss):
                    for k, h in enumerate(hs):
                        hidden[i][j][k] = Variable(hidden[i][j][k].data)
            return hidden
    else:
        assert False


def train_model(model, logging_file):
    args = model.args
    unchanged, best_dev = 0, 1000

    unroll_size = args.unroll_size
    batch_size = args.batch_size
    criterion = nn.CrossEntropyLoss(size_average=False)

    trainer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    map_to_ids = model.emb_layer.map_to_ids
    train = read_corpus(args.train, shuffle=False)
    train = create_batches(train, map_to_ids, args.batch_size, cuda=args.gpu)
    dev = read_corpus(args.dev)
    dev = create_batches(dev, map_to_ids, 1, cuda=args.gpu)
    test = read_corpus(args.test)
    test = create_batches(test, map_to_ids, 1, cuda=args.gpu)
    reduced_model_path = ""
    for epoch in range(args.max_epoch):
        
        start_time = time.time()

        N = (len(train[0]) - 1) // unroll_size + 1
        hidden = model.init_hidden(batch_size)
        total_loss, cur_loss = 0.0, 0.0

        for i in range(N):

            model.train()
            x = train[0][i*unroll_size:(i+1)*unroll_size]
            y = train[1][i*unroll_size:(i+1)*unroll_size].view(-1)

            x, y = Variable(x), Variable(y)
            model.zero_grad()
            output, hidden = model(x, hidden)
            hidden = repackage_hidden(args, hidden)
            assert x.size(1) == batch_size
            criterion_loss = criterion(output, y) / x.size(1)

            # to add the sparsifying regularizer
            
            if args.sparsity_type == "none":
                reg_loss = criterion_loss
                regularization_term = 0
            else:
                regularization_groups = train_classifier.get_regularization_groups(model, args)
                regularization_term = regularization_groups.sum()
                
                if args.reg_strength_multiple_of_loss and args.reg_strength == 0:
                    args.reg_strength = criterion_loss.data[0]*args.reg_strength_multiple_of_loss/regularization_term.data[0]

                if args.prox_step:
                    reg_loss = criterion_loss
                else:
                    reg_loss = criterion_loss + args.reg_strength * regularization_term
            loss = reg_loss
            
            loss.backward()
            if math.isnan(loss.data[0]) or math.isinf(loss.data[0]):
                print_and_log("nan/inf loss encoutered in training.", logging_file)
                sys.exit(0)
                return
            total_loss += loss.data[0] / x.size(0)
            cur_loss += loss.data[0] / x.size(0)
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.grad is not None:
                    if args.weight_decay > 0:
                        p.data.mul_(1.0 - args.weight_decay)
                    p.data.add_(-args.lr, p.grad.data)

            if (i + 1) % args.eval_ite == 0:
                #dev_ppl = eval_model(model, dev)
                dev_ppl = 9999
                print_and_log("| Epoch={} | ite={} | lr={:.4f} | train_ppl={:.2f} | dev_ppl={:.2f} |"
                                 "\n".format(
                    epoch,
                    i+1,
                    trainer.defaults["lr"],
                    np.exp(cur_loss / args.eval_ite),
                    dev_ppl
                                 ), logging_file)
                model.print_pnorm()

                sys.stdout.flush()

                cur_loss = 0.0

        train_ppl = np.exp(total_loss/N)
        # extracting the learned structure to compute its dev result
        if args.sparsity_type == "states" and True:

            model_dev_ppl = eval_model(model, dev)
            
            new_model, new_d_out = save_learned_structure.extract_learned_structure(model, args, epoch)
            if new_model is not None:
                print_and_log("size of extracted structure: {}\n".format(new_d_out), logging_file)
                new_model_dev_ppl = eval_model(new_model, dev)
                print_and_log("dev learned structure:{}, full structure {}\n".format(round(new_model_dev_ppl, 3),
                                                                                     round(model_dev_ppl, 3)), logging_file)
                dev_ppl = new_model_dev_ppl
            else:
                dev_ppl = model_dev_ppl
        else:
            dev_ppl = eval_model(model, dev)

        print_and_log("-" * 89 + "\n", logging_file)
        print_and_log("| End of epoch {} | lr={:.4f} | train_ppl={:.2f} | dev_ppl={:.2f} |"
                         "[{:.2f}m] |\n".format(
            epoch,
            trainer.defaults["lr"],
            train_ppl,
            dev_ppl,
            (time.time() - start_time) / 60.0
                         ), logging_file)
        print_and_log("-" * 89 + "\n", logging_file)
        model.print_pnorm()
        sys.stdout.flush()

        if dev_ppl < best_dev:
            unchanged = 0
            best_dev = dev_ppl
            start_time = time.time()
            
            if args.sparsity_type == "states":
                if reduced_model_path != "":
                    save_learned_structure.remove_old(reduced_model_path)
                # new_model is defined only if args.sparsity_type == "states" and an extracted model is not None
                model_test_ppl = eval_model(model, test)
                
                if new_model is not None:
                    new_model_test_ppl = eval_model(new_model, test)
                    reduced_model_path = save_learned_structure.get_model_filepath(args, new_d_out)
                    args.reduced_model_path = reduced_model_path
                    args.new_d_out = new_d_out
                    torch.save(new_model.state_dict(), reduced_model_path)
                else:
                    new_model_test_ppl = 999999
                    
                print_and_log("test learned structure:{}, full structure {}\n".format(round(new_model_test_ppl, 3),
                                                                                    round(model_test_ppl, 3)), logging_file)

                test_ppl = new_model_test_ppl
            else:
                test_ppl = eval_model(model, test)
                torch.save(model.state_dict(), args.logging_dir + args.filename + "_model.pth")
            print_and_log("\t[eval]  test_ppl={:.2f}\t[{:.2f}m]\n".format(
                test_ppl,
                (time.time() - start_time) / 60.0
            ), logging_file)
            sys.stdout.flush()

                
        else:
            unchanged += 1
        if args.lr_decay_epoch > 0 and epoch >= args.lr_decay_epoch:
            args.lr *= args.lr_decay
        if unchanged >= args.patience:
            print_and_log("Reached " + str(args.patience)
                             + " iterations without improving dev loss. Reducing learning rate.", logging_file)
            args.lr /= 2
            unchanged = 0
        trainer.defaults["lr"] = args.lr
        print_and_log("\n", logging_file)
    return


def eval_model(model, valid):
    model.eval()
    args = model.args
    total_loss = 0.0
    unroll_size = model.args.unroll_size
    criterion = nn.CrossEntropyLoss(size_average=False)
    hidden = model.init_hidden(1)
    N = (len(valid[0])-1)//unroll_size + 1
    for i in range(N):
        x = valid[0][i*unroll_size:(i+1)*unroll_size]
        y = valid[1][i*unroll_size:(i+1)*unroll_size].view(-1)
        x, y = Variable(x, volatile=True), Variable(y)
        output, hidden = model(x, hidden)
        hidden = repackage_hidden(args, hidden)
        loss = criterion(output, y)
        if math.isnan(loss.data[0]) or math.isinf(loss.data[0]):
            print("nan/inf loss encoutered in dev.")
            sys.exit(0)
            return
        total_loss += loss.data[0]
    avg_loss = total_loss / valid[1].numel()
    ppl = np.exp(avg_loss)
    return ppl

def update_environment_variables(args):
    if 'PATTERN' in os.environ:
        args.pattern = os.environ['PATTERN']
       
    if 'D_OUT' in os.environ:
        args.d_out = os.environ['D_OUT']

    if 'SEED' in os.environ:
        args.seed = int(os.environ['SEED'])

    if 'LEARNED_STRUCTURE' in os.environ:
        args.learned_structure = os.environ['LEARNED_STRUCTURE']

    if 'SEMIRING' in os.environ:
        args.semiring = os.environ['SEMIRING']

    if 'EMB_SIZE' in os.environ:
        args.emb_size = int(os.environ['EMB_SIZE'])

    if 'IN_OUT_DROPOUT' in os.environ:
        in_out_dropout = float(os.environ['IN_OUT_DROPOUT'])
        args.input_dropout = in_out_dropout
        args.output_dropout = in_out_dropout


def main(args):
    generate_filename(args)
    logging_file = train_classifier.init_logging(args)

    torch.manual_seed(args.seed)
    train = read_corpus(args.train, shuffle=False)
    model = Model(train, args)
    if args.fine_tune:
        if args.gpu:
            state_dict = torch.load(args.reduced_model_path)
        else:
            state_dict = torch.load(args.reduced_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)

    model.logging_file = logging_file
    if args.gpu:
        model.cuda()
    print_and_log("vocab size: {}\n".format(
        model.emb_layer.n_V
    ), logging_file)
    num_params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    # if args.model == "rrnn":
        # num_in = args.depth * (2 * args.d)
        # num_params = num_params - num_in
    
    print_and_log("num of parameters: {}\n".format(num_params), logging_file)
    sys.stdout.flush()
    model.print_pnorm()
    print_and_log("\n", logging_file)

    train_model(model, logging_file)
    return

def train_and_finetune(args):
    # only use lr_decay if not using regularizer
    lr_decay_epoch = args.lr_decay_epoch
    args.lr_decay_epoch = args.max_epoch + 1
    main(args)

    args.lr_decay_epoch = lr_decay_epoch
    args.fine_tune = True
    args.d_out = args.new_d_out
    args.pattern = "1-gram,2-gram,3-gram,4-gram;1-gram,2-gram,3-gram,4-gram"
    args.sparsity_type = "none"
    args.reg_strength = 0
    args.learned_structure = "l1-states-learned"
    main(args)

def print_and_log(string, logging_file):
    sys.stdout.write(string)
    logging_file.write(string)
    logging_file.flush()

def generate_filename(args):
    if args.sparsity_type == "none" and args.learned_structure:
        sparsity_name = args.learned_structure
    else:
        sparsity_name = args.sparsity_type

    name = "layers={}_lr={:.3E}_dout={}_indrop={:.4f}_outdrop={:.4f}_drop={:.4f}_wdecay={:.2E}_clip={:.2f}_pattern={}_sparsity={}".format(
        args.depth, args.lr, args.d_out, args.input_dropout, args.output_dropout,
        args.dropout, args.weight_decay, args.clip_grad, args.pattern, sparsity_name)
    if args.reg_strength > 0:
        name += "_regstr={:.3E}".format(args.reg_strength)
    if args.filename_suffix != "":
        name += args.filename_suffix
    if args.filename_prefix != "":
        name = args.filename_prefix + "_" + name
    if not args.gpu:
        name = name + "_cpu"
    if args.semiring == 'max_plus':
        name = name + "_mp"
    elif args.semiring == 'max_times':
        name = name + "_mt"
    args.filename = name

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler="resolve")
    argparser.add_argument("--seed", type=int, default=31415)
    argparser.add_argument("--model", type=str, default="rrnn")
    argparser.add_argument("--semiring", type=str, default="plus_times")
    argparser.add_argument("--depth", type=int, default=3)
    argparser.add_argument("--pattern", type=str, default="1-gram,2-gram,3-gram,4-gram;1-gram,2-gram,3-gram,4-gram;1-gram,2-gram,3-gram,4-gram")
    argparser.add_argument("--d_out", type=str, default="0,4,3,0;8,0,0,4;2,3,3,0")
    argparser.add_argument("--emb_size", type=int, default=65)
    argparser.add_argument("--num_mlp_layer", type=int, default=2)
    argparser.add_argument("--use_rho", type=str2bool, default=False)
    argparser.add_argument("--use_layer_norm", type=str2bool, default=False)
    argparser.add_argument("--use_output_gate", type=str2bool, default=True)
    argparser.add_argument("--activation", type=str, default="tanh")
    argparser.add_argument("--train", type=str, required=True, help="training file")
    argparser.add_argument("--dev", type=str, required=True, help="dev file")
    argparser.add_argument("--test", type=str, required=True, help="test file")
    argparser.add_argument("--batch_size", "--batch", type=int, default=32)
    argparser.add_argument("--unroll_size", type=int, default=35)
    argparser.add_argument("--max_epoch", type=int, default=300)
    argparser.add_argument("--input_dropout", type=float, default=0.0,
        help="dropout of word embeddings")
    argparser.add_argument("--output_dropout", type=float, default=0.0,
        help="dropout of softmax output")
    argparser.add_argument("--dropout", type=float, default=0.0,
        help="dropout intra RNN layers")
    argparser.add_argument("--rnn_dropout", type=float, default=0.0,
        help="dropout of RNN layers")
    argparser.add_argument("--lr", type=float, default=1.0)
    argparser.add_argument("--lr_decay", type=float, default=0.98)
    argparser.add_argument("--lr_decay_epoch", type=int, default=0)
    argparser.add_argument("--weight_decay", type=float, default=1e-6)
    argparser.add_argument("--clip_grad", type=float, default=5.)
    argparser.add_argument("--gpu", type=str2bool, default=False)
    argparser.add_argument("--eval_ite", type=int, default=1000)
    argparser.add_argument("--patience", type=int, default=30)
    argparser.add_argument("--sparsity_type", type=str, default="none")
    argparser.add_argument("--logging_dir", type=str, default="./logging/")
    argparser.add_argument("--filename_prefix", type=str, default="")
    argparser.add_argument("--filename_suffix", type=str, default="")
    argparser.add_argument("--reg_strength", type=float, default=0)
    argparser.add_argument("--loaded_embedding", type=bool, default=False)
    argparser.add_argument("--loaded_data", type=bool, default=False)
    argparser.add_argument("--reg_strength_multiple_of_loss", type=bool, default=False)
    argparser.add_argument("--prox_step", type=bool, default=False)
    argparser.add_argument("--learned_structure", help="Learned structure",
                           type=str, default="l1-states-learned")
    argparser.add_argument("--fine_tune", type=bool, default=False)

    args = argparser.parse_args()
    args.language_modeling = True
    sys.stdout.flush()
    update_environment_variables(args)
    print(args)

    if args.sparsity_type == "states":
        train_and_finetune(args)
    else:
        main(args)
