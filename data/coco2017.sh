# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "[ERROR] gdown not found. Please install it with:"
    echo "        pip install gdown"
    exit 1
fi

# Check if unzip is installed
if ! command -v unzip &> /dev/null; then
    echo "[ERROR] unzip not found. Please install it with:"
    echo "        sudo apt install unzip"
    exit 1
fi

cd "$(dirname "$0")"

mkdir coco2017
cd coco2017

wget -c http://images.cocodataset.org/zips/train2017.zip
echo "Extracting train2017.zip"
unzip -qq train2017.zip
rm train2017.zip

wget -c http://images.cocodataset.org/zips/val2017.zip
echo "Extracting val2017.zip"
unzip -qq val2017.zip
rm val2017.zip

wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
echo "Extracting annotations_trainval2017.zip"
unzip -qq annotations_trainval2017.zip
rm annotations_trainval2017.zip

cd annotations
find . -type f \
    -not -wholename ./instances_train2017.json \
    -not -wholename ./instances_val2017.json \
    -delete
    
echo "Add minitrain annotations"
gdown https://drive.google.com/uc?id=1lezhgY4M_Ag13w0dEzQ7x_zQ_w0ohjin

echo "DONE."
