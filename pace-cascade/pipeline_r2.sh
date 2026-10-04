#!/bin/bash
# resumable second-revision pipeline: every stage is skipped if its output exists
cd "$(dirname "$0")/code"; L=../logs/pipeline.log; R=../results
echo "START $(date)" >> $L
[ -f $R/r2_mlp.json ] || { python3 run_r2_mlp.py > ../logs/r2_mlp.log 2>&1; echo "r2_mlp $? $(date)" >> $L; }
for s in 0 1 2; do [ -f $R/deep_models/vit4_fmnist_seed$s.npz ] || { python3 deepnets.py --model vit4 --seeds $s >> ../logs/train_vit4.log 2>&1; echo "vit seed$s $? $(date)" >> $L; }; done
[ -f $R/r2_vit4.json ] || { python3 run_r2_deep.py --model vit4 > ../logs/r2_vit4.log 2>&1; echo "r2_vit4 $? $(date)" >> $L; }
for s in 0 1 2; do [ -f $R/deep_models/resnet8_cifar10_seed$s.npz ] || { python3 deepnets.py --model resnet8 --seeds $s >> ../logs/train_resnet8.log 2>&1; echo "resnet seed$s $? $(date)" >> $L; }; done
[ -f $R/r2_resnet8.json ] || { python3 run_r2_deep.py --model resnet8 > ../logs/r2_resnet8.log 2>&1; echo "r2_resnet8 $? $(date)" >> $L; }
python3 run_r2_regimes.py > ../logs/r2_regimes.log 2>&1; echo "regimes $? $(date)" >> $L
python3 run_r2_interp.py > ../logs/r2_interp.log 2>&1;   echo "interp $? $(date)" >> $L
python3 make_figs_r2.py > ../logs/figs_r2.log 2>&1;      echo "figs $? $(date)" >> $L
echo "ALL DONE $(date)" >> $L
