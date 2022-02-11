vals=($(seq -23 -1 -25))

## now loop through the above array
for i in ${vals[@]}
do
   echo "$i"
   python useb_dense_retriever.py --device cuda:2 --modelname training_nli_v2_EleutherAI-gpt-neo-1.3B-2022-01-11_12-02-29 --method mean --layeridx $i
done
