vals=($(seq -1 -1 -13))

## now loop through the above array
for i in ${vals[@]}
do
   echo "$i"
   python useb_dense_retriever.py --modelname SBERT-base-nli-v2-cust --method mean --layeridx $i
done
