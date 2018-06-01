python eval_image_classifier.py \
  --checkpoint_path=/home/zhijie.huang/github/data/TMD/train_set \
  --eval_dir=/home/zhijie.huang/github/data/TMD/validation \
  --dataset_name=chinese_font \
  --dataset_split_name=validation \
  --dataset_dir=/home/zhijie.huang/github/data/TMD \
  --batch_size=32 \
  --model_name=densenet


  python eval_image_classifier.py \
  --checkpoint_path=/home/zhijie.huang/github/data/TMD/train_set_alexnet_v2 \
  --eval_dir=/home/zhijie.huang/github/data/TMD/validation \
  --dataset_name=chinese_font \
  --dataset_split_name=validation \
  --dataset_dir=/home/zhijie.huang/github/data/TMD \
  --batch_size=32 \
  --model_name=alexnet_v2