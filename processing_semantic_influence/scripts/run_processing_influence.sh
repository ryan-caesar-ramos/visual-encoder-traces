EMBEDDINGS_DIR=$1
DEVICE=$2
OUTPUT_DIR=$3
PROCESSING=$4
K=$5
MODEL=$6
VARIANT=$7

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

for model_and_variant in "${MODELS_AND_VARIANTS[@]}"; do
	while IFS=',' read -r model variant; do
		if [[ ( "$MODEL" == "$model" || -z "$MODEL" ) && \
		( "$VARIANT" == "$variant" || -z "$VARIANT" )]]; then
			sh scripts/processing_influence.sh $EMBEDDINGS_DIR $DEVICE $OUTPUT_DIR "$PROCESSING" $K $model $variant
		fi
	done <<< $model_and_variant
done
