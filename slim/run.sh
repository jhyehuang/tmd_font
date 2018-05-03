python download_and_convert_data.py --dataset_name=chinese_font --dataset_dir=/home/zhijie.huang/github/data/TMD

python ./datasets/convert_test_data.py --dataset_name=chinese_font --dataset_dir=/home/zhijie.huang/github/data/TMD

#第一次训练
python -u train_image_classifier.py \
    --dataset_name=chinese_font \
    --dataset_dir=/home/zhijie.huang/github/data/TMD \
    --model_name=densenet \
    --checkpoint_path=/home/zhijie.huang/github/data/TMD/train_set/ \
    --train_dir=/home/zhijie.huang/github/data/TMD/train_set \
    --learning_rate=0.088 \
    --ignore_missing_vars=True \
    --batch_size=32 \
    --max_number_of_steps=100000 \
    --learning_rate_decay_type=fixed \
    --optimizer=rmsprop \
    --weight_decay=0.00004

#第二次训练
python -u train_image_classifier.py \
    --dataset_name=chinese_font \
    --dataset_dir=/home/zhijie.huang/github/data/TMD \
    --model_name=densenet \
    --checkpoint_path=/home/zhijie.huang/github/data/TMD/train_set/ \
    --train_dir=/home/zhijie.huang/github/data/TMD/train_set \
    --learning_rate=0.001 \
    --ignore_missing_vars=True \
    --batch_size=32 \
    --max_number_of_steps=50000 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=60 \
    --save_summaries_secs=60 \
    --log_every_n_steps=100 \
    --optimizer=rmsprop \
    --weight_decay=0.00004
    
#第一次训练
python -u train_image_classifier.py \
    --dataset_name=chinese_font \
    --dataset_dir=/home/zhijie.huang/github/data/TMD \
    --model_name=inception_v4 \
    --train_dir=/home/zhijie.huang/github/data/TMD/train_set_v4 \
    --learning_rate=0.088 \
    --ignore_missing_vars=True \
    --batch_size=32 \
    --max_number_of_steps=100000 \
    --learning_rate_decay_type=fixed \
    --optimizer=adam \
    --weight_decay=0.00004