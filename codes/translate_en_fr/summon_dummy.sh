src_file=/path/to/processed/wmt/train.en
tgt_file=/path/to/processed/wmt/train.fr
tot_lines=200000

CUDA_VISIBLE_DEVICES=0 \
python run_pretrain.py $src_file $tgt_file.0 0 $tot_lines &
CUDA_VISIBLE_DEVICES=0 \
python run_pretrain.py $src_file $tgt_file.200000 200000 $tot_lines &
CUDA_VISIBLE_DEVICES=0 \
python run_pretrain.py $src_file $tgt_file.400000 400000 $tot_lines &
CUDA_VISIBLE_DEVICES=0 \
python run_pretrain.py $src_file $tgt_file.600000 600000 $tot_lines &

CUDA_VISIBLE_DEVICES=1 \
python run_pretrain.py $src_file $tgt_file.800000 800000 $tot_lines &
CUDA_VISIBLE_DEVICES=1 \
python run_pretrain.py $src_file $tgt_file.1000000 1000000 $tot_lines &

CUDA_VISIBLE_DEVICES=4 \
python run_pretrain.py $src_file $tgt_file.1200000 1200000 $tot_lines &
CUDA_VISIBLE_DEVICES=5 \
python run_pretrain.py $src_file $tgt_file.1400000 1400000 $tot_lines &
CUDA_VISIBLE_DEVICES=6 \
python run_pretrain.py $src_file $tgt_file.1600000 1600000 $tot_lines &
CUDA_VISIBLE_DEVICES=7 \
python run_pretrain.py $src_file $tgt_file.1800000 1800000 $tot_lines &
CUDA_VISIBLE_DEVICES=7 \
python run_pretrain.py $src_file $tgt_file.2000000 2000000 $tot_lines &

CUDA_VISIBLE_DEVICES=4 \
python run_pretrain.py $src_file $tgt_file.2200000 2200000 $tot_lines &
CUDA_VISIBLE_DEVICES=4 \
python run_pretrain.py $src_file $tgt_file.2400000 2400000 $tot_lines &
CUDA_VISIBLE_DEVICES=4 \
python run_pretrain.py $src_file $tgt_file.2600000 2600000 $tot_lines &
CUDA_VISIBLE_DEVICES=4 \
python run_pretrain.py $src_file $tgt_file.2800000 2800000 $tot_lines &
CUDA_VISIBLE_DEVICES=4 \
python run_pretrain.py $src_file $tgt_file.3000000 3000000 $tot_lines &

CUDA_VISIBLE_DEVICES=5 \
python run_pretrain.py $src_file $tgt_file.3200000 3200000 $tot_lines &
CUDA_VISIBLE_DEVICES=5 \
python run_pretrain.py $src_file $tgt_file.3400000 3400000 $tot_lines &
CUDA_VISIBLE_DEVICES=5 \
python run_pretrain.py $src_file $tgt_file.3600000 3600000 $tot_lines &
CUDA_VISIBLE_DEVICES=5 \
python run_pretrain.py $src_file $tgt_file.3800000 3800000 $tot_lines &
CUDA_VISIBLE_DEVICES=5 \
python run_pretrain.py $src_file $tgt_file.4000000 4000000 $tot_lines &

CUDA_VISIBLE_DEVICES=7 \
python run_pretrain.py $src_file $tgt_file.4200000 4200000 $tot_lines &
CUDA_VISIBLE_DEVICES=7 \
python run_pretrain.py $src_file $tgt_file.4400000 4400000 $tot_lines &

src_file=/path/to/processed/wmt/newstest2013.en
tgt_file=/path/to/processed/wmt/newstest2013.fr.fromen
tot_lines=10000

CUDA_VISIBLE_DEVICES=5 \
python run_pretrain.py $src_file $tgt_file 0 $tot_lines &

src_file=/path/to/processed/wmt/newstest2014.en
tgt_file=/path/to/processed/wmt/newstest2014.fr.fromen

CUDA_VISIBLE_DEVICES=6 \
python run_pretrain.py $src_file $tgt_file 0 $tot_lines &
