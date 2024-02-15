python3 main.py --model_name bert-base-uncased \
                --task_name mrpc \
                --ckpt_dir ckpts/bert-base-uncased/mrpc \
                --constraint 0.7 \
                --seed 0 \
                --output my_outputs/bert-base-uncased/mrpc/mac/0.7/prune_all_d2_stand_squared \