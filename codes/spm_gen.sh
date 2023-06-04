set -ex

for lang in file.txt; do
for pref in /path/to/processed/yelp; do

cat $pref/${lang}.txt | python thumt_gen/scripts/spm.py > $pref/${lang}.spm.txt &

done

done
