python download_and_convert_data.py --dataset_name=chinese_font --dataset_dir=/home/zhijie.huang/github/data/TMD


python -u train_image_classifier.py \
    --dataset_name=chinese_font \
    --dataset_dir=/home/zhijie.huang/github/data/TMD \
    --model_name=inception_v4 \
 #   --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
    --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
    --train_dir=/home/zhijie.huang/github/data/TMD/train_set \
    --learning_rate=0.1 \
    --learning_rate_decay_factor=0.76\
    --num_epochs_per_decay=50 \
    --moving_average_decay=0.9999 \
    --optimizer=rmsprop \
    --ignore_missing_vars=True \
    --batch_size=32


 
    
