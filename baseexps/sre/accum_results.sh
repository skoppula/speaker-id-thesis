for model in fcn1 fcn2 cnn lcn maxout1 maxout2 dsc1 dsc2; do
    for bn in True ; do
        for reg in False True; do
            vals=`cat train_log/sentfiltNone_${model}_bn${bn}_reg${reg}_noLRSchedule/log.log | egrep "val-utt-error: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
            epoch=`cat train_log/sentfiltNone_${model}_bn${bn}_reg${reg}_noLRSchedule/log.log | egrep "Start Epoch [0-9]*" -o | cut -d ' ' -f 3 | tail -n1`
            mult=`cat train_log/sentfiltNone_${model}_bn${bn}_reg${reg}_noLRSchedule/log.log | egrep "mults': [0-9]*" -o | cut -d ' ' -f 2 | tail -n1`
            weights=`cat train_log/sentfiltNone_${model}_bn${bn}_reg${reg}_noLRSchedule/log.log | egrep "#params=[0-9]*" -o | cut -d '=' -f 2 | tail -n1`
            latest_val=`echo "$vals" | tail -n1`
            min_val=`echo "$vals" | sort -n | head -1`
            best_val_epoch=`echo "$vals" | cat -n | sort -k2,2nr | tail -n1 | cut -d$'\t' -f 1`
            echo $model $bn $reg: best val:$min_val at epch$best_val_epoch\; currval $latest_val at epch$epoch\; mults:$mult\; params:$weights
        done
    done
    echo
done
