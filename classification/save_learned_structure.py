import torch
import train_classifier
import numpy as np
import os, sys
sys.path.append("language_model")
sys.path.append("../language_model")
import train_lm


THRESHOLD = 0.1
ALLWFSA = None

def to_file(model, args, data_x, data_y, print_debug = True):
    new_model, new_d_out = extract_learned_structure(model, args)

    if new_model is None:
        return new_d_out, ""
        
    if print_debug:
        check_new_model_predicts_same(model, new_model, data_x, data_y, new_d_out, args.gpu)

    reduced_model_path = get_model_filepath(args, new_d_out)
    print("Writing model to", reduced_model_path)
    torch.save(new_model.state_dict(), reduced_model_path)

    return new_d_out, reduced_model_path

def remove_old(old_reduced_model_path):
    if old_reduced_model_path != "" and os.path.isfile(old_reduced_model_path):
        os.remove(old_reduced_model_path)

def get_model_filepath(args, d_out):
    if args.language_modeling:
        reduced_model_path = args.logging_dir + args.filename
    else:
        reduced_model_path = args.logging_dir + args.dataset + args.filename()
    reduced_model_path += "_model_learned={}.pth".format(d_out)
    return reduced_model_path

# a method used to make sure the extracted structure behaves similarly to the learned model
def check_new_model_predicts_same(model, new_model, data_x, data_y, new_d_out, gpu):
    # can manually look at feats vs new_model_feats, should be close (and identical for max-length WFSAs)
    #if check == "manually check features from wfsa":
    print("New_d_out={}".format(new_d_out))
    if True:
        cur_x = (data_x[0])
        model_wfsakeep_pred = predict_one_example(model, ALLWFSA, cur_x)


        if gpu:
            indices_in_new_model = torch.autograd.Variable(torch.arange(ALLWFSA.shape[0]).type(torch.cuda.LongTensor))
        else:
            indices_in_new_model = torch.autograd.Variable(torch.arange(ALLWFSA.shape[0]).type(torch.LongTensor))

        new_model_pred = predict_one_example(new_model, indices_in_new_model, cur_x)
        
        # the features which didn't make it into the smaller model:
        model_indices_not_in_new = [int(x) for x in torch.arange(24) if not x in ALLWFSA.data]
        if gpu:
            model_indices_not_in_new = torch.autograd.Variable(torch.cuda.LongTensor(model_indices_not_in_new))
        else:
            model_indices_not_in_new = torch.autograd.Variable(torch.LongTensor(model_indices_not_in_new))
        
        model_wfsadiscard_pred = predict_one_example(model, model_indices_not_in_new, cur_x, add_bias = False)


        # DEBUG stuff here
        model_feat = encoder_fwd(model, cur_x)
        model_feat = model.drop(model_feat)
        selected_feats = torch.index_select(model_feat, 1, ALLWFSA)[0,:]

        new_model_feat = encoder_fwd(new_model, cur_x)
        new_model_feat = model.drop(new_model_feat)
        
        # this shows that the 14th WFSA (seen in ALLWFSA) has the largest gap, of 0.2152, at epoch 37
        new_model_feat[0,:] - selected_feats
        
        print(model_wfsakeep_pred, new_model_pred,model_wfsadiscard_pred)
    if True:
        predict_all_train(model, new_model, data_x)
    if True:
        compare_err(model, new_model, data_x, data_y, new_d_out)

def predict_one_example(model, indices, cur_x, add_bias = True):
    if len(indices.shape) == 0:
        return "all wfsas were included in learned structure (though some may have become shorte)"

    model_feat = encoder_fwd(model, cur_x)
    model_feat = model.drop(model_feat)

    # i suspect this has problems when selected_feats is empty, because model_indices_not_in_new is empty
    selected_feats = torch.index_select(model_feat, 1, indices)[0,:]
    # to see what the contribution of the existing wfsas is, and what the contribution is of the ones that were removed

    if add_bias:
        model_b = model.output_layer.bias
    else:
        model_b = 0
    selected_weights = torch.index_select(model.output_layer.weight, 1, indices)
    #selected_weights = torch.index_select(model_w, 1, indices)
    return selected_weights.matmul(selected_feats) + model_b
    

        
def compare_err(model, new_model, data_x, data_y, new_d_out):
    model_err = round(train_classifier.eval_model(None, model, data_x, data_y), 4)
    new_model_err = round(train_classifier.eval_model(None, new_model, data_x, data_y), 4)
    print("difference: {}, model err: {}, extracted structure model err: {}, extracted structure: {}".format(
        round(new_model_err - model_err, 4), model_err, new_model_err, new_d_out))

# a method used to make sure the extracted structure behaves similarly to the learned model
def predict_all_train(model, new_model, data_x):
    model.eval()
    new_model.eval()
    total_examples = 0
    total_same_pred = 0
    for i in range(len(data_x)):
        cur_x = (data_x[i])
        total_same_pred += sum(new_model(cur_x).data.max(1)[1] == model(cur_x).data.max(1)[1])
        total_examples += cur_x[0].shape[0]
    print("total: {}, same pred: {}, frac: {}".format(total_examples, total_same_pred, round(total_same_pred * 1.0 / total_examples, 4)))
    #print("same pred: {}".format(total_same_pred))
    #assert total_same_pred * 1.0 / total_examples > .90

# a method used to make sure the extracted structure behaves similarly to the learned model
def encoder_fwd(model, cur_x):
    model.eval()
    emb_fwd = model.emb_layer(cur_x)
    emb_fwd = model.drop(emb_fwd)

    out_fwd, hidden_fwd, _ = model.encoder(emb_fwd)
    batch, length = emb_fwd.size(-2), emb_fwd.size(0)
    out_fwd = out_fwd.view(length, batch, 1, -1)
    feat = out_fwd[-1,:,0,:]
    return feat

def extract_learned_structure(model, args, epoch = 0):
    states = train_classifier.get_states_weights(model, args)
    num_wfsas = sum([int(one_size) for one_size in args.d_out.split(";")[0].split(",")])
    layers = len(model.encoder.rnn_lst)
    num_ngrams_per_layer = []
    num_of_each_ngram_per_layer = []
    for i in range(layers):
        #cur_layer_states = states[:,i*num_wfsas:(i+1)*num_wfsas,:]
        cur_layer_states = states[i]
        num_ngrams, num_of_each_ngram = find_num_ngrams(cur_layer_states, num_wfsas)
   
        if max(num_ngrams) == -1:
            return None, "0,0,0,0"

        num_ngrams_per_layer.append(num_ngrams)
        num_of_each_ngram_per_layer.append(num_of_each_ngram)

    new_model, new_d_out = create_new_model(num_of_each_ngram_per_layer, args, model, layers)

    #if new_d_out != "0,0,0,710;0,0,0,710":
    #    import pdb; pdb.set_trace()
    
    all_wfsa_indices_by_layer = []
    # updating each layer's weights
    for layer in range(layers):
        update_new_model_weights(model, new_model, num_ngrams_per_layer[layer],
                                                    args, layer, all_wfsa_indices_by_layer)
    if not args.language_modeling:
        # updating the Linear layer at the end
        update_linear_output_layer(model, new_model, num_ngrams_per_layer[layer], torch.cat(all_wfsa_indices_by_layer[-1]))
    else:
        update_projection_layer(model, new_model, torch.cat(all_wfsa_indices_by_layer[-1]))
        update_embedding_layer(model, new_model)
        
    return new_model, new_d_out

def update_embedding_layer(model, new_model):
    new_model.emb_layer = model.emb_layer
    new_model.output_layer = model.output_layer

def update_projection_layer(model, new_model, all_wfsa_indices):
    model_weights = model.hidden.weight
    model_bias = model.hidden.bias.data

    new_model_weights = new_model.hidden.weight.data
    new_model_bias = new_model.hidden.bias.data

    cur_model_weights = torch.index_select(model_weights, 1, all_wfsa_indices).data

    # DEBUG
    if not new_model_weights.shape == cur_model_weights.shape:
        import pdb; pdb.set_trace()
    
    new_model_weights.copy_(cur_model_weights)
    new_model_bias.copy_(model_bias)
    
    # DEBUG
    global ALLWFSA
    ALLWFSA = all_wfsa_indices


def update_linear_output_layer(model, new_model, num_ngrams, all_wfsa_indices):
    model_weights = model.output_layer.weight
    model_bias = model.output_layer.bias.data

    new_model_weights = new_model.output_layer.weight.data
    new_model_bias = new_model.output_layer.bias.data

    cur_model_weights = torch.index_select(model_weights, 1, all_wfsa_indices).data

    # DEBUG
    if not new_model_weights.shape == cur_model_weights.shape:
        import pdb; pdb.set_trace()
    
    
    new_model_weights.copy_(cur_model_weights)
    new_model_bias.copy_(model_bias)
    
    
    # DEBUG
    global ALLWFSA
    ALLWFSA = all_wfsa_indices


# all weights are either "model" weights or "new_model" weights, as denoted in the name of the local variable
def update_new_model_weights(model, new_model, num_ngrams, args, layer, all_wfsa_indices_by_layer):
    #embed_dim = model.emb_layer.emb_size
    
    # need to extract the weights that are non-zero, and if any are non-zero for a particular wfsa,
    # should also extract the final column, which contains the output gate weights.
    num_edges_in_wfsa = model.encoder.rnn_lst[layer].cells[0].k
    uses_output_gate = model.encoder.rnn_lst[layer].cells[0].ngram * 2 + 1 == model.encoder.rnn_lst[layer].cells[0].k
    num_wfsas = sum([int(one_size) for one_size in args.d_out.split(";")[layer].split(",")])
    
    reshaped_model_weights = model.encoder.rnn_lst[layer].cells[0].weight.view(-1, num_wfsas, num_edges_in_wfsa)

    cur_cell_num = 0
    all_wfsa_indices = []
    
    for i in range(int(num_edges_in_wfsa/2)):
        # if there are no ngrams of this length, continue
        if sum(num_ngrams == i) == 0:
            #import pdb; pdb.set_trace()
            continue
        # to get the indices of the wfsas of length i in reshaped_model_weights
        if args.gpu:
            wfsa_indices = torch.autograd.Variable(torch.cuda.LongTensor(np.where(num_ngrams == i)[0]))
        else:
            wfsa_indices = torch.autograd.Variable(torch.LongTensor(np.where(num_ngrams == i)[0]))

        update_mult_weights(model, reshaped_model_weights, new_model, wfsa_indices, args, i, cur_cell_num,
                            layer, all_wfsa_indices_by_layer)
        update_bias_weights(num_edges_in_wfsa, num_wfsas, model, new_model, wfsa_indices, args, i, cur_cell_num, layer)
        all_wfsa_indices.append(wfsa_indices)
        cur_cell_num += 1

    assert not model.encoder.bidirectional
    assert not args.use_rho

    all_wfsa_indices_by_layer.append(all_wfsa_indices)    
    
def update_bias_weights(num_edges_in_wfsa, num_wfsas, model, new_model, wfsa_indices, args, i, cur_cell_num, layer):
    # it looks like we don't actually use the first half of the first dim of the bias anywhere.

    uses_output_gate = model.encoder.rnn_lst[layer].cells[0].ngram * 2 + 1 == model.encoder.rnn_lst[layer].cells[0].k
    model_bias = model.encoder.rnn_lst[layer].cells[0].bias.view(num_edges_in_wfsa, 1, num_wfsas)
    cur_model_bias_full = torch.index_select(model_bias, 2, wfsa_indices)

    # to get the parts of the bias that are actually used
    model_start_index = int(cur_model_bias_full.shape[0]/2)
    # [model_start_index - (i+1) : model_start_index + (i+1)] is the middle set of params from this matrix
    cur_model_bias = cur_model_bias_full[model_start_index - (i+1) : model_start_index + (i+1), ...]

    if uses_output_gate:
        cur_model_bias = torch.cat((cur_model_bias, cur_model_bias_full[-1,...].unsqueeze(0)),0)
    
    cur_new_model_bias = new_model.encoder.rnn_lst[layer].cells[cur_cell_num].bias.view((i+1)*2 + uses_output_gate, 1, wfsa_indices.shape[0])
    
    cur_new_model_bias_data = cur_new_model_bias.data
    cur_model_bias_data = cur_model_bias.data
    
    cur_new_model_bias_data.add_(cur_model_bias_data)
        
        
# updates the multiplicative weights in new_model to be the same as in model, for the patterns of length i+1
def update_mult_weights(model, reshaped_model_weights, new_model, wfsa_indices, args, i, cur_cell_num, layer, all_wfsa_indices_by_layer):

    uses_output_gate = model.encoder.rnn_lst[layer].cells[0].ngram * 2 + 1 == model.encoder.rnn_lst[layer].cells[0].k
    cur_model_weights_full = torch.index_select(reshaped_model_weights, 1, wfsa_indices)
    if layer != 0:
        cur_model_weights_full = torch.index_select(cur_model_weights_full, 0, torch.cat(all_wfsa_indices_by_layer[-1]))
    # to get only the non-zero states
    cur_model_weights = cur_model_weights_full[:,:,0:(i+1)*2]
    if uses_output_gate:
        cur_model_weights = torch.cat((cur_model_weights, cur_model_weights_full[:,:,-1].unsqueeze(2)), 2)
    
    cur_new_model_weights = new_model.encoder.rnn_lst[layer].cells[cur_cell_num].weight
    cur_new_model_weights = cur_new_model_weights.view(cur_model_weights.shape[0],
                                                       cur_model_weights.shape[1], (i+1)*2 + uses_output_gate)
        
    cur_new_model_weights_data = cur_new_model_weights.data
    cur_model_weights_data = cur_model_weights.data
    
    cur_new_model_weights_data.add_(-cur_new_model_weights_data)
    cur_new_model_weights_data.add_(cur_model_weights_data)
        
        
def create_new_model(num_of_each_ngram, args, model, layers):
    # to store the current d_out and pattern
    tmp_d_out = args.d_out
    tmp_pattern = args.pattern

    new_d_out = ""
    new_pattern = ""
    # semi-colon separated d_outs, one for each layer
    for layer in range(layers):

        # to generate the new learned d_out
        new_d_out += ";"
        cur_num_of_each_ngram = num_of_each_ngram[layer]
        
        for i in range(len(cur_num_of_each_ngram)):
            new_d_out += "{},".format(cur_num_of_each_ngram[i])
        new_d_out = new_d_out[:-1]

        new_pattern += ";1-gram,2-gram,3-gram,4-gram"

    # to remove the initial semicolon
    new_d_out = new_d_out[1:]
    new_pattern = new_pattern[1:]
    
    # setting the new d_out and pattern in args
    args.d_out = new_d_out
    args.pattern = new_pattern

    # creating the new model
    if args.language_modeling:
        new_model = train_lm.Model(model.emb_layer.word2id.keys(), args)
    else:
        new_model = train_classifier.Model(args, model.emb_layer, model.output_layer.out_features)
    if args.gpu:
        new_model.cuda()

    # putting the d_out and pattern back in args
    args.d_out = tmp_d_out
    args.pattern = tmp_pattern
    return new_model, new_d_out
    
    
def find_num_ngrams(states, num_wfsas):
    # a list which stores the n-gram of each wfsa (e.g. 0,1,2 etc.) 
    num_ngrams = []
    
    num_states = states.shape[2]

    # to find the largest state which is above the threshold
    for i in range(num_wfsas):
        cur_max_state = -1
        prev_group_norm = -1
        for j in range(num_states):

            cur_group = states[:,i,j].data

            if cur_group.norm(2) > THRESHOLD and cur_max_state == j - 1:
                # and prev_group_norm * .1 < cur_group.norm(2)
                cur_max_state = j
            prev_group_norm = cur_group.norm(2)
        num_ngrams.append(cur_max_state)

    num_ngrams = np.asarray(num_ngrams)
    num_of_each_ngram = []
    for i in range(num_states):
        num_of_each_ngram.append(sum(num_ngrams == i))

    return num_ngrams, num_of_each_ngram
    
    
