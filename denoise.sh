
usage()
{
	echo "Usage: $0 input_video_name output_video_name"

	exit 1
}

[ "$1" = "" ] && usage
[ "$2" = "" ] && usage

TEMP_INPUT_DIR=/tmp/fastdvdnet_input
TEMP_OUTPUT_DIR=/tmp/fastdvdnet_output

mkdir -p $TEMP_INPUT_DIR
mkdir -p $TEMP_OUTPUT_DIR


ffcoder --decode $1 ${TEMP_INPUT_DIR}/%3d.png

python test_fastdvdnet.py \
	--test_path ${TEMP_INPUT_DIR} \
	--noise_sigma 10 \
	--save_path ${TEMP_OUTPUT_DIR}

ffcoder --encode ${TEMP_OUTPUT_DIR}/%3d.png $2

rm -rf $TEMP_OUTPUT_DIR
rm -rf $TEMP_INPUT_DIR
