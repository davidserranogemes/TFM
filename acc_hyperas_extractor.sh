cd
cd master/TFM/


VALUE="$1_acc.txt"



cat -A $1| grep "Epoch\|val_acc:" | sed 's/.*\(Epoch.*\)/\1/' | sed 's/.*\(- [0-9].*\)/\1/ '| sed 'N;s/\n/ /' | sed 's/\(Epoch\).\([0-9]\+\)\/[0-9]\+\(.*\)/\1\2\3/' | tr -s " " "-" |  awk -F'-' '{print $1 $6 $10}' | sed 's/\(.*\)\([0-9]\.[0-9]\+\)/\1 \2/' | sed 's/\(.*\)\([0-9]\.[0-9]\+\) \(.*\)/\1 \2 \3/' | tr -s '$' " " > $VALUE
