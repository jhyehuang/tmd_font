#第一次训练
python -u drive/Colaboratory/tmd_font/slim/train_image_classifier.py \
    --dataset_name=chinese_font \
    --dataset_dir=drive/Colaboratory/train_data \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=drive/Colaboratory/train_data/chkpoint/ \
    --train_dir=drive/Colaboratory/tmd_font/output \
    --learning_rate=0.01 \
    --ignore_missing_vars=True \
    --batch_size=32 \
    --max_number_of_steps=20000 \
    --learning_rate_decay_type=fixed \
    --optimizer=rmsprop \
    --weight_decay=0.00004

#第二次训练
#python -u drive/Colaboratory/tmd_font/slim/train_image_classifier.py \
#    --dataset_name=chinese_font \
#    --dataset_dir=drive/Colaboratory/tmd_font/train_data \
#    --model_name=densenet \
#    --checkpoint_path=drive/Colaboratory/tmd_font/train_data/chkpoint/ \
#    --train_dir=drive/Colaboratory/tmd_font/output \
#    --learning_rate=0.001 \
#    --ignore_missing_vars=True \
#    --batch_size=32 \
#    --max_number_of_steps=5000 \
#    --learning_rate_decay_type=fixed \
#    --optimizer=rmsprop \
#    --weight_decay=0.00004
    

!python -u drive/Colaboratory/tmd_font/slim/train_image_classifier.py --dataset_name=chinese_font --dataset_dir=drive/Colaboratory/train_data --model_name=inception_resnet_v2 --checkpoint_path=drive/Colaboratory/train_data/chkpoint/inception_resnet_v2_2016_08_30.ckpt --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits  --train_dir=drive/Colaboratory/tmd_font/output/ --learning_rate=0.01  --batch_size=24 --max_number_of_steps=20000 --learning_rate_decay_type=fixed --optimizer=adam --weight_decay=0.00004



--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits 