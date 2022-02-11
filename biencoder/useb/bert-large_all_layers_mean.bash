vals=($(seq -1 -1 -24))

## now loop through the above array
for i in ${vals[@]}
do
   echo "$i"
   python useb_dense_retriever.py --modelname bert-large-uncased-cust --method mean --layeridx $i
done
