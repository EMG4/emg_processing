#!/bin/bash
#==============================================================================
# Author: Carl Larsson
# Description: Script for testing all features in feature extraction
# Date: 2023-05-09
#==============================================================================
ctr0=0
ctr1=1

cp feature/feature_list.py feature/feature_list_0.py

NAME=feature/feature_list

while true ; do
        sed -e '0,/"use":"yes"/s/"use":"yes"/"use":"no"/' \
            -e '0,/"use":""/s/"use":""/"use":"yes"/' ${NAME}_$ctr0.py > ${NAME}_$ctr1.py
        diff ${NAME}_$ctr0.py ${NAME}_$ctr1.py
        status=$?
        if [ $status == 0 ]; then
                echo "Break for ${NAME}_$ctr0.txt ${NAME}_$ctr1.txt"
                rm ${NAME}_$ctr0.py ${NAME}_$ctr1.py
                break
        fi
        cp -f ${NAME}_$ctr1.py $NAME.py
        python3 main.py -f 150hz.txt --hz 150 --rmlp 1
        echo "${NAME}_$ctr0.py ${NAME}_$ctr1.py"
        rm ${NAME}_$ctr0.py
        ((ctr0++))
        ((ctr1++))
done
