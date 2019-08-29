import load_learned_structure
from experiment_params import ExperimentParams
import train_classifier
import numpy as np
import time, os
import save_learned_structure

LR_LOWER_BOUND = 7*10**-3
LR_UPPER_BOUND = .5

BERT_LR_LOWER_BOUND = 7*10**-5
BERT_LR_UPPER_BOUND = .5*10**-2


def search_reg_str_entropy(cur_assignments, kwargs):
    starting_reg_str = kwargs["reg_strength"]
    file_base = "/home/jessedd/projects/rational-recurrences/classification/logging/" + kwargs["dataset"]    
    found_small_enough_reg_str = False
    # first search by checking that after 5 epochs, more than half aren't above .9
    kwargs["max_epoch"] = 1
    counter = 0
    rho_bound = .99
    while not found_small_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_structure.entropy_rhos(
            file_base + args.filename() + ".txt", rho_bound)
        print("fraction under {}: {}".format(rho_bound,frac_under_pointnine))
        print("")
        if frac_under_pointnine < .25:
            kwargs["reg_strength"] = kwargs["reg_strength"] / 2.0
            if kwargs["reg_strength"] < 10**-7:
                kwargs["reg_strength"] = starting_reg_str
                return counter, "too_big_lr"
        else:
            found_small_enough_reg_str = True

    found_large_enough_reg_str = False
    kwargs["max_epoch"] = 5
    rho_bound = .9
    while not found_large_enough_reg_str:
        counter += 1
        args = ExperimentParams(**kwargs, **cur_assignments)
        cur_valid_err, cur_test_err = train_classifier.main(args)
        
        learned_pattern, learned_d_out, frac_under_pointnine = load_learned_structure.entropy_rhos(
            file_base + args.filename() + ".txt", rho_bound)
        print("fraction under {}: {}".format(rho_bound,frac_under_pointnine))
        print("")
        if frac_under_pointnine > .25:
            kwargs["reg_strength"] = kwargs["reg_strength"] * 2.0
            if kwargs["reg_strength"] > 10**4:
                kwargs["reg_strength"] = starting_reg_str
                return counter, "too_small_lr"
        else:
            found_large_enough_reg_str = True
    # to set this back to the default
    kwargs["max_epoch"] = 500
    return counter, "okay_lr"

# ways this can fail:
# too small learning rate
# too large learning rate
# too large step size for reg strength, so it's too big then too small
def search_reg_str_l1(cur_assignments, kwargs, global_counter, distance_from_target = 10):
    # the final number of params is within this amount of target
    smallest_reg_str = 10**-9
    largest_reg_str = 10**2
    starting_reg_str = kwargs["reg_strength"]
    found_good_reg_str = False
    too_small = False
    too_large = False
    counter = 0
    reg_str_growth_rate = 2.0
    reduced_model_path = ""
    
    while not found_good_reg_str:
        # deleting models which aren't going to be used

        save_learned_structure.remove_old(reduced_model_path)

        # if more than 25 regularization strengths have been tried, throw out hparam assignment and resample
        if counter > 25:
            kwargs["reg_strength"] = starting_reg_str
            return counter, "bad_hparams", cur_valid_err, learned_d_out, reduced_model_path

        counter += 1
        args = ExperimentParams(counter = global_counter, **kwargs, **cur_assignments)
        cur_valid_err, learned_d_out, reduced_model_path = train_classifier.main(args)
        
        num_params = sum([int(learned_d_out.split(",")[i]) * (i+1) for i in range(len(learned_d_out.split(",")))])
        
        if num_params < kwargs["reg_goal_params"] - distance_from_target:
            if too_large:
                # reduce size of steps for reg strength
                reg_str_growth_rate = (reg_str_growth_rate + 1)/2.0
                too_large = False
            too_small = True
            kwargs["reg_strength"] = kwargs["reg_strength"] / reg_str_growth_rate
            if kwargs["reg_strength"] < smallest_reg_str:
                kwargs["reg_strength"] = starting_reg_str
                return counter, "too_small_lr", cur_valid_err, learned_d_out, reduced_model_path
        elif num_params > kwargs["reg_goal_params"] + distance_from_target:
            if too_small:
                # reduce size of steps for reg strength
                reg_str_growth_rate = (reg_str_growth_rate + 1)/2.0
                too_small = False
            too_large = True
            kwargs["reg_strength"] = kwargs["reg_strength"] * reg_str_growth_rate

            if kwargs["reg_strength"] > largest_reg_str:
                kwargs["reg_strength"] = starting_reg_str
                
                # it diverged, and for some reason the weights didn't drop
                if num_params == int(args.d_out) * 4 and cur_assignments["lr"] > .1 and cur_valid_err > .3:
                    return counter, "too_big_lr", cur_valid_err, learned_d_out, reduced_model_path
                else:
                    return counter, "too_small_lr", cur_valid_err, learned_d_out, reduced_model_path
        else:
            found_good_reg_str = True

    return counter, "okay_lr", cur_valid_err, learned_d_out, reduced_model_path

# to finetune the learned structure
def get_kwargs_for_fine_tuning(kwargs, reduced_model_path, learned_d_out, pattern):
    new_kwargs = {key: value for key, value in kwargs.items()}
    new_kwargs["learned_structure"] = "l1-states-finetuning"
    new_kwargs["sparsity_type"] = "none"
    new_kwargs["fine_tune"] = True
    new_kwargs["reduced_model_path"] = reduced_model_path
    new_kwargs["reg_strength"] = 0
    new_kwargs["d_out"] = learned_d_out
    new_kwargs["pattern"] = pattern
    return new_kwargs
    
def train_k_then_l_models(k,l,counter,total_evals,start_time, logging_dir, distance_from_target, **kwargs):
    if "seed" in kwargs and kwargs["seed"] is not None:
        np.random.seed(kwargs["seed"])
        
    assert "reg_strength" in kwargs
    if "prox_step" not in kwargs:
        kwargs["prox_step"] = False
    elif kwargs["prox_step"]:
        assert False, "It's too unstable. books/all_cs_and_equal_rho/hparam_opt/structure_search/proximal_gradient too big then too small"
    assert kwargs["sparsity_type"] == "states", "setting kwargs for structure learning works only with states"
    assert "lr_patience" not in kwargs, "lr_patience is set s.t. the lr never decreases during structure learning."
    kwargs["logging_dir"] = logging_dir

    file_base = logging_dir + kwargs["dataset"]    
    best = {
        "assignment" : None,
        "valid_err" : 1,
        "learned_pattern" : None,
        "learned_d_out" : None,
        "reg_strength": None
        }

    reg_search_counters = []
    if kwargs["bert_embed"]:
        lr_lower_bound = BERT_LR_LOWER_BOUND
        lr_upper_bound = BERT_LR_UPPER_BOUND
    else:
        lr_lower_bound = LR_LOWER_BOUND
        lr_upper_bound = LR_UPPER_BOUND
    all_assignments = get_k_sorted_hparams(k, lr_lower_bound, lr_upper_bound)
    for i in range(len(all_assignments)):

        valid_assignment = False
        while not valid_assignment:
            cur_assignments = all_assignments[i]
            
            # to prevent the learning rate from decreasing during structure learning
            kwargs["lr_patience"] = 9999999
            
            if kwargs["sparsity_type"] == "rho_entropy":
                one_search_counter, lr_judgement = search_reg_str_entropy(cur_assignments, kwargs)
            elif kwargs["sparsity_type"] == "states":
                one_search_counter, lr_judgement, cur_valid_err, learned_d_out, reduced_model_path = search_reg_str_l1(
                    cur_assignments, kwargs, counter[0], distance_from_target)
                learned_pattern = "1-gram,2-gram,3-gram,4-gram"

            del kwargs["lr_patience"]
            
            reg_search_counters.append(one_search_counter)
            if lr_judgement == "okay_lr":
                valid_assignment = True
            else:
                save_learned_structure.remove_old(reduced_model_path)
                new_assignments = get_k_sorted_hparams(k-i, lr_lower_bound, lr_upper_bound, sort=False)
                all_assignments[i:len(all_assignments)] = new_assignments

                #if lr_judgement == "too_big_lr":
                #    # lower the upper bound
                #    lr_upper_bound = cur_assignments['lr']
                #    reverse = True
                #elif lr_judgement == "too_small_lr":
                #    # rase lower bound
                #    lr_lower_bound = cur_assignments['lr']
                #    reverse = False
                #else:
                #    assert False, "shouldn't be here."
                #new_assignments = get_k_sorted_hparams(k-i, lr_lower_bound, lr_upper_bound)
                #if reverse:
                #    new_assignments.reverse()
                #all_assignments[i:len(all_assignments)] = new_assignments
                

        # to fine tune the learned model
        kwargs_fine_tune = get_kwargs_for_fine_tuning(kwargs, reduced_model_path, learned_d_out, learned_pattern)
        args = ExperimentParams(counter = counter[0], **kwargs_fine_tune, **cur_assignments)
        cur_valid_err, _, _ = train_classifier.main(args)
            
        if cur_valid_err < best["valid_err"]:
            best = {
                "assignment" : cur_assignments,
                "valid_err" : cur_valid_err,
                "learned_pattern" : learned_pattern,
                "learned_d_out" : learned_d_out,
                "reg_strength": kwargs["reg_strength"]
            }

        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))

    kwargs["reg_strength"] = best["reg_strength"]
    for i in range(l):
        kwargs["lr_patience"] = 9999999
        args = ExperimentParams(counter = counter[0], filename_suffix="_{}".format(i),**kwargs, **best["assignment"])
        cur_valid_err, learned_d_out, reduced_model_path = train_classifier.main(args)
        del kwargs["lr_patience"]
        
        # to fine tune the model trained on the above line
        kwargs_fine_tune = get_kwargs_for_fine_tuning(kwargs, reduced_model_path, learned_d_out, learned_pattern)
        args = ExperimentParams(counter = counter[0], filename_suffix="_{}".format(i), **kwargs_fine_tune, **best["assignment"])
        cur_valid_err, learned_d_out, reduced_model_path = train_classifier.main(args)
        
        counter[0] = counter[0] + 1
        
    
    return best, reg_search_counters


def train_m_then_n_models(m,n,counter, total_evals,start_time,**kwargs):
    if kwargs["bert_embed"]:
        lr_lower_bound = BERT_LR_LOWER_BOUND
        lr_upper_bound = BERT_LR_UPPER_BOUND
    else:
        lr_lower_bound = LR_LOWER_BOUND
        lr_upper_bound = LR_UPPER_BOUND
    best_assignment = None
    best_valid_err = 1
    all_assignments = get_k_sorted_hparams(m, lr_lower_bound, lr_upper_bound)
    for i in range(m):
        cur_assignments = all_assignments[i]
        args = ExperimentParams(counter = counter[0], **kwargs, **cur_assignments)
        cur_valid_err, _, _ = train_classifier.main(args)
        if cur_valid_err < best_valid_err:
            best_assignment = cur_assignments
            best_valid_err = cur_valid_err
        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))

    for i in range(n):
        args = ExperimentParams(counter = counter[0], filename_suffix="_{}".format(i),**kwargs,**best_assignment)
        cur_valid_err, _, _ = train_classifier.main(args)
        counter[0] = counter[0] + 1
        print("trained {} out of {} hyperparameter assignments, so far {} seconds".format(
            counter[0],total_evals, round(time.time()-start_time, 3)))
    return best_assignment

# hparams to search over (from paper):
# clip_grad, dropout, learning rate, rnn_dropout, embed_dropout, l2 regularization (actually weight decay)
def hparam_sample(lr_bounds):
    assignments = {
        "clip_grad": np.random.uniform(1.0, 5.0),
        "dropout": np.random.uniform(0.0, 0.5),
        "rnn_dropout": np.random.uniform(0.0, 0.5),
        "embed_dropout": np.random.uniform(0.0, 0.5),
        "lr": np.exp(np.random.uniform(np.log(lr_bounds[0]), np.log(lr_bounds[1]))),
        "weight_decay": np.exp(np.random.uniform(np.log(10 ** -5), np.log(10 ** -7))),
    }

    return assignments


# orders them in increasing order of lr
def get_k_sorted_hparams(k, lr_upper_bound, lr_lower_bound, sort=True):
    all_assignments = []

    for i in range(k):
        cur = hparam_sample(lr_bounds=[lr_lower_bound, lr_upper_bound])
        all_assignments.append([cur['lr'], cur])
    if sort:
        all_assignments.sort()
    return [assignment[1] for assignment in all_assignments]
