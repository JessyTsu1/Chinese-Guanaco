CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 python ../src/train_embedding.py \
    --model_name_or_path /remote-home/rikka/law-guanaco/guanaco-33b-merged \
    --tokenizer_name_or_path /remote-home/rikka/law-guanaco/guanaco-33b-merged \
    --torch_dtype float16