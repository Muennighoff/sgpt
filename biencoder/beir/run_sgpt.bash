# Modelname to run
modelname=${1:--1}
device=${2:-cuda:0}

declare -a arr=("msmarco" "nfcorpus" "bioasq" "nq" "hotpotqa" "fiqa"
                "signal1m" "trec-news" "arguana" "webis-touche2020" "quora" "dbpedia-entity"
                "scidocs" "fever" "climate-fever" "scifact" "robust04" "cqadupstack/english" "cqadupstack/android" "cqadupstack/english" "cqadupstack/gaming"
               "cqadupstack/gis" "cqadupstack/mathematica" "cqadupstack/physics" "cqadupstack/programmers"
               "cqadupstack/stats" "cqadupstack/wordpress" "cqadupstack/webmasters" "cqadupstack/unix"
               "cqadupstack/tex" "trec-covid")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   python beir_dense_retriever.py --device $device --specb --modelname $modelname --usest --dataset $i --batchsize 16 --method weightedmean
done
