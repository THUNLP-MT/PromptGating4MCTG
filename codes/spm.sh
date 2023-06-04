set -ex

for lang in fr; do

for pref in /path/to/processed/wmt/train; do
cat $pref.$lang | python thumt/scripts/spm.py > $pref.spm.$lang &
done

done
