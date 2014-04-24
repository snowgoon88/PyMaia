#! /bin/bash 
for class in {1..3}
do
	for i in {1..50}
	do
		python trajgen.py --class "$class" dataset/"$class"_"$i"
	done
done