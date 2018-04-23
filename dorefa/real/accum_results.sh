act="False"

for quant_ends in False True; do
    echo "" > /tmp/plot.txt
    for model in lcn fcn1 fcn2 cnn; do
        tmp=""
        for w in 2 4 8 16 32; do
            basedir="train_log/${model}_w_${w}_a_32_quant_ends_${quant_ends}"
            if [ "${quant_ends}" == "True" ] && [ "$model" == "cnn" ] ; then
                basedir="${basedir}_preload"
            fi
            if [ "${quant_ends}" == "True" ] && [ "$model" == "lcn" ] ; then
                basedir="${basedir}_preload"
            fi
            if [ "${quant_ends}" == "True" ] && [ "$model" == "fcn1" ] ; then
                basedir="${basedir}_preload"
            fi
            vals=`cat ${basedir}/log.log | egrep "val-utt-error: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
            epoch=`echo "$vals" | wc -w`
            latest_val=`echo "$vals" | tail -n1`
            min_val=`echo "$vals" | sort -n | head -1`
            best_val_epoch=`echo "$vals" | cat -n | sort -k2,2nr | tail -n1 | cut -d$'\t' -f 1`
            echo $model $w $quant_ends: best val:$min_val at epch$best_val_epoch\; currval $latest_val at epch$epoch\;
            tmp="$tmp $min_val"
        done
        tmp="$(echo -e "${tmp}" | sed -e 's/^[[:space:]]*//')"
        echo $tmp >> /tmp/plot.txt
    done
    python graph.py $quant_ends $act
    echo
done

act="True"

for quant_ends in True; do
    echo "" > /tmp/plot.txt
    for model in lcn fcn1 fcn2 cnn; do
        tmp=""
        for w in 2 4 8 16 32; do
            basedir="train_log/${model}_w_${w}_a_${w}_quant_ends_${quant_ends}"
            if [ "${w}" != "2" ] || [ "$model" != "fcn2" ] ; then
                basedir="${basedir}_preload"
            fi
            if [ "${w}" == "32" ] && [ "$model" == "fcn2" ] ; then
                basedir="${basedir:0:40}"
            fi

            vals=`cat ${basedir}/log.log | egrep "val-utt-error: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
            epoch=`echo "$vals" | wc -w`
            latest_val=`echo "$vals" | tail -n1`
            min_val=`echo "$vals" | sort -n | head -1`
            best_val_epoch=`echo "$vals" | cat -n | sort -k2,2nr | tail -n1 | cut -d$'\t' -f 1`
            echo $model $w-$w $quant_ends: best val:$min_val at epch$best_val_epoch\; currval $latest_val at epch$epoch\;
            tmp="$tmp $min_val"
        done
        tmp="$(echo -e "${tmp}" | sed -e 's/^[[:space:]]*//')"
        echo $tmp >> /tmp/plot.txt
    done
    python graph.py $quant_ends $act
    echo
done
