# Modelname to run
modelname=${1:--1}
device=${2:-cuda:0}


### Define your checkpoints, several examples follow ###
# GPT125 CKPTS
declare -a arr=("15600" "31200" "46800" "62400" "78000")
# GPT1.3 CKPTS
#declare -a arr=("62398" "78000")
# GPT2.7 CKPTS
#declare -a arr=("101387" "124784" "148181" 
#                "156000" "31196" "54593" 
#                "7799" "93588" "109186" 
#                "132583" "15598" "38995" 
#                "62392" "77990" "116985" 
#                "140382" "155980" "23397" 
#                "46794" "70191" "85789")
# GPT-5.8B CKPTS
#declare -a arr=("112311"  "137269"  "174706"  
#                "237101"  "262059"  "299496"  
#                "37437"  "74874" "12479"   
#                "149748"  "187185"  "212143"     
#                "24958"   "274538"  "311975"  
#                "49916"  "87353" "124790"  
#                "162227"  "199664"  "224622"     
#                "249580"  "287017"  "311990"  
#                "62395"  "99832")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   python beir_dense_retriever.py --device $device --specb --modelname $modelname/$i --usest --dataset scifact --batchsize 32
   python beir_dense_retriever.py --device $device --specb --modelname $modelname/$i --usest --dataset nfcorpus --batchsize 32
   python beir_dense_retriever.py --device $device --specb --modelname $modelname/$i --usest --dataset fiqa --batchsize 32
   python beir_dense_retriever.py --device $device --specb --modelname $modelname/$i --usest --dataset scidocs --batchsize 32
   python beir_dense_retriever.py --device $device --specb --modelname $modelname/$i --usest --dataset arguana --batchsize 32
done
