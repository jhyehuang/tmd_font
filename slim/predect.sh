python -u eval_image_classifier.py \
  --dataset_name=chinese_font \
  --dataset_dir=/media/han/code/data \
  --dataset_split_name=train \
  --model_name=inception_v4 \
  --checkpoint_path=/media/han/code/my_train \
  --eval_dir=/media/han/code/my_eval \
  --batch_size=32 \
  --num_examples=1328
