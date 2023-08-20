CUDA_VISIBLE_DEVICES=0  python  /home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/tools/train.py \
                                --config /home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/configs/detr/detr_r50_8xb2-150e_pascal.py  \
                                --work-dir ./train_results/20230814/epoch300_train4_val2_enc3_dec6_hdim256_query150 \
                                --cfg-options this_is_added_to_cfg_keys=and_this_becomes_the_value
                                
