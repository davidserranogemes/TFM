#!/bin/bash
cd
cd master/TFM/


VALUE="$1_acc.txt"

#Dont collapse same values

#Collapse same values
cat -A $1 | grep "Epoch" | tr -dc '0-9.:\ Epoch' |tr ":" "\n" | grep "Epoch" | awk '{print $ 3 $5}' | grep "Epoch" | uniq | sed 's/^ch\(Epoch[0-9][0-9]\?\)\(0.*\)/\1: \2/' | grep "Epoch[0-9].*" | sed 's/\(Epoch\)\([0-9]*\)\(.*\)/\1 \2\3/' > $VALUE



