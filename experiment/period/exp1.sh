#! /bin/bash

i=0

for period in {10..200..10}
do
  for n in {10..50..10}
  do
    for esnSeed in 7 12 24 31 36 42 48 54 59 62
    do
      for seqSeed in 8 13 29 35 47
      do
        ./Period.py --esn data/esn$n-1-1.json \
                    --esnSeed $esnSeed \
                    --seqSeed $seqSeed \
                    --period $period \
                    --cardinal 5 \
                    --init 100 \
                    --train 1000 \
                    --test 2000 \
                    --regul 1e-3 \
                    --output exp1.out
                    
        i=$(($i+1)) 
        printf "Progress: %i%% \r" $(($i/50))
        done
    done
  done
done