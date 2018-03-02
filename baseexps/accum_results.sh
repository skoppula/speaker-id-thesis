maxe=18
for model in fcn1 fcn2 cnn lcn maxout1 maxout2 dsc1 dsc2; do
    for bn in True False; do
        for reg in False True; do
            if [ "$bn" == "False" ] && [ "$reg" == "True" ] ; then
                continue
            fi
            basedir="train_log/sentfiltNone_${model}_bn${bn}_reg${reg}_noLRSchedule"
            vals=`cat ${basedir}/log.log | egrep "val-utt-error: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
            if [ -d "${basedir}_preload" ] ; then
                vals2=`cat ${basedir}_preload/log.log | egrep "val-utt-error: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
                vals="$vals $vals2"
            fi
            epoch=`echo "$vals" | wc -w`
            latest_val=`echo "$vals" | tail -n1`
            mult=`cat train_log/sentfiltNone_${model}_bn${bn}_reg${reg}_noLRSchedule/log.log | egrep "mults': [0-9]*" -o | cut -d ' ' -f 2 | tail -n1`
            weights=`cat train_log/sentfiltNone_${model}_bn${bn}_reg${reg}_noLRSchedule/log.log | egrep "#params=[0-9]*" -o | cut -d '=' -f 2 | tail -n1`
            min_val=`echo "$vals" | sort -n | head -1`
            best_val_epoch=`echo "$vals" | cat -n | sort -k2,2nr | tail -n1 | cut -d$'\t' -f 1`
            score=`bc -l <<< "l(${min_val}*${mult}*${weights})/l(10)"`
            echo $model $bn $reg: best val:$min_val at epch$best_val_epoch\; currval $latest_val at epch$epoch\; mults:$mult\; params:$weights, err-mult-param-score: $score
        done
    done
    echo
done
