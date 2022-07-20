CUDA_VISIBLE_DEVICES=2,3 python test.py \
--model 6 \
--checkpoints ./model_234 \
--input /misc/dl001/dataset/yamashita/paris_street_view_dataset/paris_test_im \
--mask /misc/dl001/dataset/yamashita/mask_test_dataset/mask_4 \
--depth /misc/dl001/dataset/yamashita/paris_street_view_dataset/paris_test_depth_im \
--output paris_results_4