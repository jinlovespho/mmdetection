CUDA_VISIBLE_DEVICES=0  python /home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/tools/test.py \
                        --config /home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/configs/detr_sparse/detr_sparse_r50_8xb2-150e_pascal.py \
                        --checkpoint /home/kwangrok/Downloads/VS_CODE/my_github/mmdetection/tools/my_script/detr/train/train_results/20230814/epoch300_train4_val2_enc3_dec6_hdim256_query150/epoch_300.pth \
                        --work-dir ./test_results/20230820/epoch300_test_on_traindset \
                        --show-dir ./show_dir_img
