
usage()
{
	echo "Usage: $0 input_video_name output_video_name"

	exit 1
}

# [ "$1" = "" ] && usage
# [ "$2" = "" ] && usage

TEMP_INPUT_DIR=project/dataset/predict/input
TEMP_OUTPUT_DIR=project/dataset/predict/output

# mkdir -p ${TEMP_INPUT_DIR}
# mkdir -p ${TEMP_OUTPUT_DIR}

# video_coder is a bash

# video_coder --decode "$1" "${TEMP_INPUT_DIR}/%03d.png"

python test_fastdvdnet.py \
	--test_path ${TEMP_INPUT_DIR} \
	--noise_sigma 10 \
	--save_path ${TEMP_OUTPUT_DIR}

# video_coder --encode "${TEMP_OUTPUT_DIR}/%03d.png" "$2"

# rm -rf ${TEMP_OUTPUT_DIR}
# rm -rf ${TEMP_INPUT_DIR}
