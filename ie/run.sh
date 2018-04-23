#!/bin/bash -e

. cmd.sh
. path.sh

# define some stuff...
stage=6
nj=32
ubmsize=500
ivectordim=600

echo $(date)

#   Data preparation
if [ $stage -le 0 ]; then
    . local/data_prep.sh
fi

# copy-feats ark:raw_mfcc_enroll.1.ark ark,t:text.txt
# copy-vector ark:vad_enroll.1.ark ark,t:vad1.txt

#   Feature extraction: MFCCs
if [ $stage -le 1 ]; then
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $nj \
        --cmd "$train_cmd" data exp/mfcc data/mfcc
fi

#   VAD
if [ $stage -le 2 ]; then
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
        data/ exp/vad data/vad
fi


if [ $stage -le 3 ]; then
    # should be --delta-window=3 --delta-order=2
    delta_opts=`cat exp/ubm_diag_${ubmsize}/delta_opts 2>/dev/null`
    POSTVAD_DIR="data/dd_mfcc_postvad"
    if [ ! -d "$POSTVAD_DIR" ]; then
        mkdir $POSTVAD_DIR
    fi
    add-deltas $delta_opts scp:data/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:data/vad.scp ark,scp:$POSTVAD_DIR/dd_mfcc_postvad.ark,$POSTVAD_DIR/dd_mfcc_postvad.scp
fi

#   Train UBM with diagonal covariance
if [ $stage -le 4 ]; then
    sid/train_diag_ubm.sh --nj $nj --cmd "$train_cmd" \
        data/ $ubmsize exp/ubm_diag_$ubmsize
fi

#   Train UBM with full covariance (using diag. UBM as starting point)
if [ $stage -le 5 ]; then
    sid/train_full_ubm.sh --nj $nj --cmd "$train_cmd" \
        data/ exp/ubm_diag_$ubmsize exp/ubm_full_$ubmsize
fi

if [ $stage -le 6 ]; then
    sid/train_ivector_extractor.sh --nj $nj --cmd "$train_cmd" \
        --ivector_dim $ivectordim exp/ubm_full_${ubmsize}/final.ubm \
        data/ exp/ie_${ubmsize}_${ivectordim}
fi

#   Extract i-vectors for all of our data.
if [ $stage -le 7 ]; then
    sid/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
            exp/ie_${ubmsize}_${ivectordim} data/ exp/ivectors_${ubmsize}_${ivectordim}
fi

if [ $stage -le 8 ]; then
    #   Length normalize per-utterance i-vectors
    srcdir=exp/ivectors_${ubmsize}_${ivectordim}
    dir=exp/ivectors_norm_${ubmsize}_${ivectordim}
    $train_cmd JOB=1:$nj $dir/log/length_norm.JOB.log ivector-normalize-length scp:$srcdir/ivector.JOB.scp  ark,scp:$dir/ivector_norm.JOB.ark,$dir/ivector_norm.JOB.scp 
    for j in $(seq $nj); do cat $dir/ivector_norm.$j.scp; done >$dir/ivector_norm.scp
fi
