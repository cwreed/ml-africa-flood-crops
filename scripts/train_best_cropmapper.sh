#!/bin/sh
conda activate floodcrops
python -m train_model \
    --model_type cropland \
    --run_name best_crop \
    --checkpoint \
    --num_lstm_layers 2 \
    --hidden_size 128 \
    --lstm_dropout 0.03893478492878551 \
    --num_classification_layers 2 \
    --classifier_hidden_size 512 \
    --optimizer adam \
    --learning_rate 0.0009366333405034162 \
    --weight_decay 0.0001033039853916173 \
    --patience 10