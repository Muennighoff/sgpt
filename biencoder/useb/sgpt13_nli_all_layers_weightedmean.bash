vals=($(seq -1 -1 -25))

## now loop through the above array
for i in ${vals[@]}
do
   echo "$i"
   python useb_dense_retriever.py --device cuda:3 --modelname Muennighoff/SGPT-1.3B-weightedmean-nli --method weightedmean --layeridx $i
done
