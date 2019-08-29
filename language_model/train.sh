gpu_id=1
num_cpu=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export OMP_NUM_THREADS=${num_cpu}


if [ $# -lt 2 ]; then
	echo "Usage: $0 <data dir> <log dir>"
	exit -1
fi

# pattern="1-gram,2-gram,3-gram,4-gram"
# hidden_size="150,150,150,150"
# d=600
# input_dropout=0.6
# output_dropout=0.6

# pattern="1-gram,2-gram,3-gram"
# hidden_size="210,210,210"
# d=630
# input_dropout=0.6
# output_dropout=0.6

# pattern="1-gram,2-gram"
# hidden_size="330,330"
# d=660
# input_dropout=0.6
# output_dropout=0.6

pattern="4-gram;4-gram"
d_out="780;780"
#pattern="1-gram,2-gram,3-gram,4-gram;1-gram,2-gram,3-gram,4-gram"
#d_out="124,277,68,6;8,239,178,17"
#d_out="0,293,320,167;0,0,2,778"
#d_out="66,150,34,4;94,176,55,6"

emb_size=250
input_dropout=0.65
output_dropout=0.65

lr=1.0
lr_decay=0.98
lr_decay_epoch=150
activation="tanh"
batch_size=32
model="rrnn"
depth=2
input_dropout=0.6
output_dropout=0.6
dropout=0.2
rnn_dropout=0.2
use_output_gate=True
unroll_size=35
use_rho=False
max_epoch=350
weight_decay=1e-5
patience=30
gpu=True
sparsity_type=states
reg_strength=0.000085
logging_dir=$2
eval_ite=100


com="python3.6 train_lm.py --train $1/train --dev $1/dev --test $1/test \
--d_out=$d_out \
--emb_size=$emb_size \
--lr=$lr \
--lr_decay=$lr_decay \
--lr_decay_epoch=$lr_decay_epoch \
--activation=$activation \
--batch_size=$batch_size \
--model=$model \
--pattern=$pattern \
--depth=$depth \
--input_dropout=$input_dropout \
--output_dropout=$output_dropout \
--dropout=$dropout \
--rnn_dropout=$rnn_dropout \
--use_output_gate=$use_output_gate \
--unroll_size=$unroll_size \
--use_rho=$use_rho \
--max_epoch=$max_epoch \
--weight_decay=$weight_decay \
--patience=$patience \
--gpu=$gpu \
--sparsity_type=$sparsity_type \
--reg_strength=$reg_strength \
--logging_dir=$logging_dir \
--eval_ite=$eval_ite"


echo $com

$com

#nohup $com \
#> $2/${pattern}.${hidden_size}.${depth}.${input_dropout}.${output_dropout}.${dropout}.${lr}.${lr_decay}.${lr_decay_epoch} &
