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

#第一次训练
python -u train_image_classifier.py \
    --dataset_name=chinese_font \
    --dataset_dir=/home/zhijie.huang/github/data/TMD \
    --model_name=alexnet_v2 \
    --train_dir=/home/zhijie.huang/github/data/TMD/train_set_alexnet_v2 \
    --learning_rate=0.088 \
    --ignore_missing_vars=True \
    --batch_size=32 \
    --max_number_of_steps=100000 \
    --learning_rate_decay_type=fixed \
    --optimizer=adam 
    --weight_decay=0.00004


#预测
python test_image_classifier.py \
  --checkpoint_path=/home/zhijie.huang/github/data/TMD/train_set_alexnet_v2/ \
  --test_dir=/home/zhijie.huang/github/data/TMD/test_alexnet_v2 \
  --dataset_name=test_chinese_font \
  --dataset_split_name=test \
  --dataset_dir=/home/zhijie.huang/github/data/TMD \
  --batch_size=1 \
  --labels_file=/home/zhijie.huang/github/data/TMD/labels.txt \
  --model_name=alexnet_v2