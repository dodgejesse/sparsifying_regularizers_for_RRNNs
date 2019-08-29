
# to run a regularized experiment
#python run_random_search_experiments.py --logging_dir logging/ --filename_prefix "all_cs_and_equal_rho/hparam_opt/structure_search/add_reg_term_to_loss/DEBUG/" --max_epoch 5 --k 1 --l 1 --m 1 --n 1 --base_data_dir ~/data/ --dataset amazon_categories/original_mix/ --pattern "4-gram" --d_out 24 --gpu


# to run a baseline:
python run_random_search_experiments.py --logging_dir logging/ --filename_prefix "all_cs_and_equal_rho/hparam_opt/" --base_data_dir ~/data/ --dataset "bert/amazon_categories/kitchen_&_housewares/" --gpu --bert_embed True --baseline --batch_size 4 --reg_goal_params False --sparsity_type "none" --reg_strength 0

