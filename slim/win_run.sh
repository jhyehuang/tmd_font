python download_and_convert_data.py --dataset_name=chinese_font --dataset_dir=D:/GitHub/data/TMD

python ./datasets/convert_test_data.py --dataset_name=chinese_font --dataset_dir=D:/GitHub/data/TMD


python -u train_image_classifier.py \
    --dataset_name=chinese_font \
    --dataset_dir=D:/GitHub/data/TMD \
    --model_name=inception_v4 \
    --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
    --trainable_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits \
    --train_dir=D:/GitHub/data/TMD/train_set \
    --learning_rate=0.001 \
    --learning_rate_decay_factor=0.76\
    --num_epochs_per_decay=50 \
    --moving_average_decay=0.9999 \
    --optimizer=adam \
    --ignore_missing_vars=True \
    --batch_size=32


  
python test.py ^
  --checkpoint_path=D:/GitHub/data/TMD/train_set/model.ckpt-18753 ^
  --test_dir=D:/GitHub/data/TMD/test ^
  --dataset_name=test_chinese_font ^
  --dataset_split_name=test ^
  --dataset_dir=D:/GitHub/data/TMD ^
  --batch_size=1 ^
  --labels_file=D:/GitHub/data/TMD/labels.txt ^
  --model_name=densenet
  
python test_image_classifier.py ^
  --checkpoint_path=D:/GitHub/data/TMD/train_set/model.ckpt-18753 ^
  --test_dir=D:/GitHub/data/TMD/test ^
  --dataset_name=test_chinese_font ^
  --dataset_split_name=test ^
  --dataset_dir=D:/GitHub/data/TMD ^
  --batch_size=1 ^
  --labels_file=D:/GitHub/data/TMD/labels.txt ^
  --model_name=densenet