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
   python beir_openai_embeddings_batched_parallel.py --datapath ../datasets/ --engine curie --dataset "$i"
done
