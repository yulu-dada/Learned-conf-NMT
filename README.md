# Our code is implemented on Fairseq.

# Env: Fairseq-0.9, Pytorch-1.6
# Code structure:
- mycode : Learned Confidence Estimate
- mycode-ls: Confidence-based Label Smoothing

# Train
```
user_dir=./mycode/
data_bin=../data-bin-chen/
model_dir=./models/chen-conf
export CUDA_VISIBLE_DEVICES=0,1,2,3
nohup fairseq-train $data_bin \
        --user-dir $user_dir --criterion my_label_smoothed_cross_entropy --task my_translation_task --arch my_arch \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr-scheduler inverse_sqrt \
        --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.0005 --min-lr 1e-09 \
        --weight-decay 0.0 --label-smoothing 0.1 \
        --max-tokens 4096 --no-progress-bar --max-update 150000 \
        --log-interval 100 --save-interval-updates 1000 --keep-interval-updates 10 --save-interval 10000 --seed 1111 \
        --ddp-backend no_c10d \
        --dropout 0.3 \
        -s ch -t en --save-dir $model_dir \
        --c-loss-weight 30 \ # λ_0 in equation(8)
        --beta 30000 > log.train-chen-conf &   # β_0 in equation (8)
```

1. For all QE tasks and out-of-domain data detection tasks, we set λ_0=30, β_0=30000.
2. For IWSLT De-En in noisy data identification tasks, we set λ_0=30, β_0=6000.
3. For confidence-based LS task, the settings of λ_0 and β_0 are in Appendix. 

# Inference
```fairseq-generate $test_path --max-tokens 4200 --user-dir $user_dir --criterion my_label_smoothed_cross_entropy --task translation --path $model_path -s ch -t en --beam 4 --remove-bpe > $tgt.txt &
```