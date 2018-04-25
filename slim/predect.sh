python -u eval_image_classifier.py ^
  --dataset_name=chinese_font ^
  --dataset_dir=D:\GitHub\data\TMD ^
  --dataset_split_name=validation ^
  --model_name=densenet ^
  --checkpoint_path=D:\GitHub\data\TMD\train_set ^
  --eval_dir=D:\GitHub\data\TMD\train_set\my_eval ^
  --batch_size=32 ^
  --num_examples=1328
