for model in fcn1 fcn2 cnn lcn maxout1 maxout2 dsc1 dsc2; do
    for bn in True False; do
        for reg in True False; do
            vals=`cat train_log/sentfiltNone_${model}_bn${bn}_reg${reg}/log.log | egrep "val-utt-error: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
            epoch=`cat train_log/sentfiltNone_${model}_bn${bn}_reg${reg}/log.log | egrep "Start Epoch [0-9]*" -o | cut -d ' ' -f 3 | tail -n1`
            latest_val=`echo "$vals" | tail -n1`
            min_val=`echo "$vals" | sort -n | head -1`
            best_val_epoch=`echo "$vals" | cat -n | sort -k2,2nr | tail -n1 | cut -d$'\t' -f 1`
            echo $model $bn $reg: $min_val, epoch $best_val_epoch $latest_val, epoch $epoch
        done
    done
    echo
done