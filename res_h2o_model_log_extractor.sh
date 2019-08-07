#!/bin/bash
cd
cd master/TFM/


VALUE="$1_mean_class_error.txt"


#cat -A $1 | grep "New leader" | tr -dc '0-9. \n' | awk '{print $3}' > $VALUE

cat $1 | grep "^DeepLearning_grid" | awk '{print $1 $2}' | tr '_' ' ' | awk '{print $8}' | sed 's/\([0-9]\?[0-9]\)\(0\.[0-9]\+\)/\1 \2/' | sort -n >$VALUE

