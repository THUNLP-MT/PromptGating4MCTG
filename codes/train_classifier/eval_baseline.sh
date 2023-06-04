prefix=/home/lzj/lzj/plug4MSG/MultiControl/res/tense/predict
output_loc=/home/lzj/lzj/plug4MSG/MultiControl/res/tense/acc.log

for i in 0 1
do
    for j in 0 1 2
    do

echo [$i,$j,0] >> $output_loc
CUDA_VISIBLE_DEVICES=5 \
python eval.py --file_loc ${prefix}_${i}${j}0.txt --specify [$i,$j] >> $output_loc
ref_ten=/home/lzj/lzj/plug4MSG/data/yelp/infer/past_label.375.txt
python /home/lzj/lzj/plug4MSG/simple_tense_detector/tense_detector_t.py -input ${prefix}_${i}${j}0.txt -output ${prefix}_${i}${j}0.tense -url http://127.0.0.1:8092/
python /home/lzj/lzj/plug4MSG/exp_tense/calc_tense_acc.py ${prefix}_${i}${j}0.tense $ref_ten ${prefix}_${i}${j}0.log

echo [$i,$j,1] >> $output_loc
CUDA_VISIBLE_DEVICES=5 \
python eval.py --file_loc ${prefix}_${i}${j}1.txt --specify [$i,$j] >> $output_loc
ref_ten=/home/lzj/lzj/plug4MSG/data/yelp/infer/present_label.375.txt
python /home/lzj/lzj/plug4MSG/simple_tense_detector/tense_detector_t.py -input ${prefix}_${i}${j}1.txt -output ${prefix}_${i}${j}1.tense -url http://127.0.0.1:8092/
python /home/lzj/lzj/plug4MSG/exp_tense/calc_tense_acc.py ${prefix}_${i}${j}1.tense $ref_ten ${prefix}_${i}${j}1.log

echo [$i,$j,2] >> $output_loc
CUDA_VISIBLE_DEVICES=5 \
python eval.py --file_loc ${prefix}_${i}${j}2.txt --specify [$i,$j] >> $output_loc
ref_ten=/home/lzj/lzj/plug4MSG/data/yelp/infer/future_label.375.txt
python /home/lzj/lzj/plug4MSG/simple_tense_detector/tense_detector_t.py -input ${prefix}_${i}${j}2.txt -output ${prefix}_${i}${j}2.tense -url http://127.0.0.1:8092/
python /home/lzj/lzj/plug4MSG/exp_tense/calc_tense_acc.py ${prefix}_${i}${j}2.tense $ref_ten ${prefix}_${i}${j}2.log


        done;
    done;
done;
