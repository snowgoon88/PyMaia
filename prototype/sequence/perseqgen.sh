#! /bin/bash

for i in {10..200..10}
do
	python perseqgen.py -p $i -o ../data/mem/per5/per$i
done