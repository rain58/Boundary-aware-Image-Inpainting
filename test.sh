CUDA_VISIBLE_DEVICES=2,3 python test.py \
--model 6 \
--checkpoints ./checkpoints \
--input test_image_dataset_path \
--mask test_mask_dataset_path \
--depth test_depth_dataset_path \
--output result
