for dataset in  Computers Squirrel 
do
  echo $dataset
  for net in GAT
  do
    for Original_ot in ot
    do
      for lr in 0.001 0.002 0.01 0.05
      do
       for weight_decay in 0.0 0.0005
        do
         for lambda in 0.1 1.0 10.0 1000.0
         do
           python training_para.py --device 3 --lr $lr --weight_decay $weight_decay --lambda $lambda --dataset  $dataset --net $net --Original_ot $Original_ot --train_rate 0.6 --val_rate 0.2 --dprate 0.5
         done
        done
      done
    done
  done
done