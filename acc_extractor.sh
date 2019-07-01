#!/bin/bash
cd
cd master/TFM/logs


VALUE="acc_$1"

#Dont collapse same values
#cat -A logs/$1.txt | grep "Epoch"  | tr -dc '0-9.: ' | tr -s " " "\n" | grep :$ | tr ":" " "

#Collapse same values
cat -A $1 | grep "Epoch" | tr -dc '0-9.:\ Epoch' | tr -s " " "\n" | grep ":$\|Epoch" |tr ":" "\t " | sed 's/^ch\(.*\)/\1/' | tr "\n" " " | tr "\t" "\n" | sed 's/^ \(.*\)/\1/' | uniq > $VALUE



