CUDA_VISIBLE_DEVICES=0  python  /home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/tools/train.py \
                                --config /home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/configs/conditional_detr/conditional-detr_r50_8xb2-50e_pascal.py  \
                                --work-dir ./train_results/20230813/epoch50_train4_val4_enc6_dec6_hdim256_query300
                                