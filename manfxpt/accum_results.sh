for model in fcn1 fcn2 cnn lcn maxout1 maxout2 dsc1 dsc2; do
    for bn in True; do
        for reg in True False; do
            for bit in 4 8 16 32; do
                val=`cat train_log/${model}_bn_${bn}_reg_${reg}_wbit_${bit}_abit_${bit}_bnbit_${bit}_biasbit_${bit}_overflow_0.01/log.log | egrep "Final cumulative utt accuracy" | cut -d ' ' -f 11`
                echo $model $bn $reg $bit: val:$val 
            done
        done
    done
    echo
done
