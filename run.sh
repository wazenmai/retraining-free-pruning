python3 main.py --model_name bert-base-uncased \
                --task_name sst2 \
                --ckpt_dir ckpts/bert-base-uncased/sst2 \
                --constraint 0.6 \
                --seed 0 \
                --output my_outputs/bert-base-uncased/sst2/mac/0.6/test \
                --d2_norm z-score