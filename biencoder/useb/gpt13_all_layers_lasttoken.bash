vals=($(seq -1 -1 -25))

## now loop through the above array
for i in ${vals[@]}
do
   echo "$i"
   python useb_dense_retriever.py --modelname gpt13 --method lasttoken --layeridx $i
done
