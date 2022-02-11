declare -a arr=("msmarco" "trec-covid" "nfcorpus" "bioasq" "nq" "hotpotqa" "fiqa"
                "signal1m" "trec-news" "arguana" "webis-touche2020" "quora" "dbpedia-entity"
                "scidocs" "fever" "climate-fever" "scifact" "robust04" "cqadupstack/english" "cqadupstack/android" "cqadupstack/english" "cqadupstack/gaming"
               "cqadupstack/gis" "cqadupstack/mathematica" "cqadupstack/physics" "cqadupstack/programmers"
               "cqadupstack/stats" "cqadupstack/wordpress" "cqadupstack/webmasters" "cqadupstack/unix"
               "cqadupstack/tex")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   python bert_dense_retriever.py --datapath ../datasets/ --modelname bert-base-uncased --usest --revision b96743c503420c0858ad23fca994e670844c6c05 --dataset "$i"
done
