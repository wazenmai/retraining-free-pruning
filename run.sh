TASK=sst2
CONSTRAINT=0.7
IMPORTANCE=r2
NAME=prune_all_$IMPORTANCE

python3 main.py --model_name bert-base-uncased \
                --task_name $TASK \
                --ckpt_dir ckpts/bert-base-uncased/$TASK \
                --constraint $CONSTRAINT \
                --seed 0 \
                --importance $IMPORTANCE \
                --output my_outputs/bert-base-uncased/$TASK/mac/$CONSTRAINT/$NAME \