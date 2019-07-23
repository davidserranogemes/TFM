#!/bin/bash
cd
cd master/TFM/


VALUE="$1_mean_class_error.txt"


cat -A $1 | grep -e "DL_random_grid" | grep "^[0-9]" | awk '{print $4 $5}' | sed 's/\(.*\)\([0-9]\.[0-9]*\)/\1 \2/' > $VALUE
