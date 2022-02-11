vals=($(seq -1 -1 -12))

## now loop through the above array
for i in ${vals[@]}
do
   echo "$i"
   python useb_dense_retriever.py --modelname bert-base-uncased-cust --method mean --layeridx $i
done
