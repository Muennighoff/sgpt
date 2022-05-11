vals=($(seq -1 -1 -13))

## now loop through the above array
for i in ${vals[@]}
do
   echo "$i"
   python useb_dense_retriever.py --modelname ANONYMIZED/SGPT-125M-scratchmean-nli --method mean --layeridx $i
done
