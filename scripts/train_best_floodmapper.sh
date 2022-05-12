#!/bin/sh
conda activate floodcrops
python -m train_model \
    --model_type flood
    --run_name best_flood
    --checkpoint
    --multi_headed
    --perm_water_proportion 0
    --alpha 0.14350606450765968
    --num_lstm_layers 3
    --hidden_size 128
    --lstm_dropout 0.06673831657497757
    --num_classification_layers 3
    --classifier_hidden_size 1024
    --optimizer adam
    --learning_rate 0.000942043065006378
    --weight_decay 0.00004373501604610397

    