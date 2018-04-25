python download_and_convert_data.py --dataset_name=chinese_font --dataset_dir=/home/zhijie.huang/github/data/TMD

python ./datasets/convert_test_data.py --dataset_name=chinese_font --dataset_dir=/home/zhijie.huang/github/data/TMD


python -u train_image_classifier.py \
    --dataset_name=chinese_font \
    --dataset_dir=/home/zhijie.huang/github/data/TMD \
    --model_name=densenet \
    --checkpoint_path=/home/zhijie.huang/github/data/TMD/train_set/model.ckpt-14448 \
    --train_dir=/home/zhijie.huang/github/data/TMD/train_set \
    --learning_rate=0.001 \
    --optimizer=rmsprop \
    --ignore_missing_vars=True \
    --batch_size=32


 
    
