SPLIT_DATA_JSON_DIR=$1
EMBEDDINGS_DIR=$2
DEVICE=$3
OUTPUT_DIR_PREFIX=$4
MODEL=$5
VARIANT=$6
TARGET_ATTRIBUTE=$7

MODELS_AND_VARIANTS=(
	"clip,ViT-B-16"
	"clip,ViT-B-32"
	"clip,ViT-L-14"
	"clip,ViT-L-14@336px"
	"clip,RN50"
	"clip,RN101"
	"clip,RN50x4"
	"clip,RN50x16"
	"clip,RN50x64"
	"openclip,ViT-B-16-laion2B"
	"openclip,ViT-B-32-laion2B"
	"openclip,ViT-L-14-laion2B"
	"openclip,ViT-H-14-laion2B"
	"openclip,ViT-g-14-laion2B"
	"openclip,ViT-B-16-DataComp.XL"
	"openclip,ViT-B-32-DataComp.XL"
	"openclip,ViT-L-14-DataComp.XL"
	"openclip,convnext_base"
	"openclip,convnext_large"
	"openclip,convnext_xxlarge"
	"siglip,base_256"
	"siglip,large_256"
	"siglip2,base_256"
	"siglip2,large_256"
	"vit,base_patch16_224"
	"vit,base_patch32_224"
	"vit,large_patch16_224"
	"vit,large_patch32_224"
	"vit,huge_patch14_224"
	"resnet,50"
	"resnet,101"
	"convnext,tiny"
	"convnext,base"
	"convnext,large"
	"convnext,xlarge"
	"dino,vits8"
	"dino,vits16"
	"dino,vitb8"
	"dino,vitb16"
	"dino,resnet50"
	"dinov2,vits14_reg"
	"dinov2,vitb14_reg"
	"dinov2,vitl14_reg"
	"dinov2,vitg14_reg"
	"mocov3,vit_small"
	"mocov3,vit_base"
	"mocov3,resnet50"
)

TARGET_ATTIRIBUTES=(
	"Make"
	"Aperture"
)

for model_and_variant in "${MODELS_AND_VARIANTS[@]}"; do
	while IFS=',' read -r model variant; do
		for attribute in "${TARGET_ATTIRIBUTES[@]}"; do
			if [[ ( "$MODEL" == "$model" || -z "$MODEL" ) && \
			( "$VARIANT" == "$variant" || -z "$VARIANT" )  && \
			( "$TARGET_ATTRIBUTE" == "$attribute" || -z "$TARGET_ATTRIBUTE" )]]; then
				CUDA_VISIBLE_DEVICES=$DEVICE python acquisition_prediction.py \
					--model $model \
					--variant $variant \
					--target_attribute "$attribute" \
					--split_data_json_dir $SPLIT_DATA_JSON_DIR \
					--embeddings_dir $EMBEDDINGS_DIR \
					--features_norm="l2" \
					--clf_type="logreg_torch" \
					--seed=42 \
					--output_dir "$OUTPUT_DIR_PREFIX/model=${model}_variant=${variant}_target_attribute=${attribute}/"
			fi
		done
	done <<< $model_and_variant
done
