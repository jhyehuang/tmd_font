python test_image_classifier.py \
  --checkpoint_path=/home/zhijie.huang/github/data/TMD/train_set/model.ckpt-18753 \
  --test_dir=/home/zhijie.huang/github/data/TMD/test \
  --dataset_name=test_chinese_font \
  --dataset_split_name=test \
  --dataset_dir=/home/zhijie.huang/github/data/TMD \
  --batch_size=1 \
  --labels_file=/home/zhijie.huang/github/data/TMD/labels.txt \
  --model_name=densenet
  