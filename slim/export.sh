python -u export_inference_graph.py \
    --model_name=densenet \
    --output_file=./chinese_font_densenet.pb \
    --dataset_name=chinese_font \
    --dataset_dir=/home/zhijie.huang/github/data/TMD
      
python -u /home/zhijie.huang/anaconda3/lib/python3.6/site-packages/tensorflow/python/tools/freeze_graph.py \
      --input_graph=chinese_font_densenet.pb \
      --input_checkpoint=/home/zhijie.huang/github/data/TMD/train_set/model.ckpt-18753 \
      --output_graph=/home/zhijie.huang/github/data/TMD/validation/chinese_font_densenet_v1.0.pb \
      --input_binary=True \
      --output_node_names=densenet/DenseNet/predictions
      
    cp /media/han/code/data/labels.txt ./my_inception_v4_freeze.label 


