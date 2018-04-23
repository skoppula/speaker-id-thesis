# fcn1_w_4_a_32_quant_ends_False/

for model in fcn2 cnn; do
    for quant_ends in False True; do
        basedir="train_log/${model}_a_32_quant_ends_${quant_ends}"
        basedir="${basedir}_preload"
        vals=`cat ${basedir}/log.log | egrep "val-utt-error: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
        epoch=`echo "$vals" | wc -w`

        latest_val=`echo "$vals" | tail -n1`
        min_val=`echo "$vals" | sort -n | head -1`
        best_val_epoch=`echo "$vals" | cat -n | sort -k2,2nr | tail -n1 | cut -d$'\t' -f 1`

        echo $model $quant_ends: best val:$min_val at epch$best_val_epoch\; currval $latest_val at epch$epoch\;
    done
    echo
done

mkdir -p figs

echo "Sparsity:"
for model in fcn2 cnn; do
    for quant_ends in False True; do
        basedir="train_log/${model}_a_32_quant_ends_${quant_ends}"
        basedir="${basedir}_preload"

        percentn1=`cat ${basedir}/log.log | egrep "linear1/W_0_percent_n: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
        percentp1=`cat ${basedir}/log.log | egrep "linear1/W_0_percent_p: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
        sparsity1=`cat ${basedir}/log.log | egrep "linear1/W_0_sparsity: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`

        sp1=`echo $sparsity1 | rev | cut -d ' ' -f1 | rev`
        echo $sparsity1 > /tmp/sparsitycounter
        echo $percentn1 >> /tmp/sparsitycounter
        echo $percentp1 >> /tmp/sparsitycounter

        python graph_sparsity.py figs/${model}_${quant_ends}_1.png

        percentn2=`cat ${basedir}/log.log | egrep "linear2/W_0_percent_n: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
        percentp2=`cat ${basedir}/log.log | egrep "linear2/W_0_percent_p: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`
        sparsity2=`cat ${basedir}/log.log | egrep "linear2/W_0_sparsity: [0-9]*.[0-9]*" -o | cut -d ' ' -f 2`

        sp2=`echo $sparsity2 | rev | cut -d ' ' -f1 | rev`
        echo $model $quant_ends: $sp1 $sp2
        echo $sparsity2 > /tmp/sparsitycounter
        echo $percentn2 >> /tmp/sparsitycounter
        echo $percentp2 >> /tmp/sparsitycounter

        python graph_sparsity.py figs/${model}_${quant_ends}_2.png

    done
    echo
done
