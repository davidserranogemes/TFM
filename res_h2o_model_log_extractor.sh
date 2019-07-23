#!/bin/bash
cd
cd master/TFM/


VALUE="$1_mean_class_error.txt"


cat -A $1 | grep "New leader" | tr -dc '0-9. \n' | awk '{print $3}' > $VALUE
