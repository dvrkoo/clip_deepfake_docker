
python train.py \
    --name=clip_vitl14_mediaeval_ftval4k_randomfc--1 \
    --wang2020_data_path "./datasets/" \
    --data_mode mediaeval_val  \
    --data_label val \
    --arch=CLIP:ViT-L/14  \
    --fix_backbone

# --data_aug
# --fix_backbone
# --reseme_ckpt "./pretrained_weights/fc_weights.pth"
