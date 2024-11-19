echo "starting donwload !\n"
cd SageMaker/mnt/efs
mkdir MHIST && cd MHIST
echo "download annotations.csv"
wget https://xxxx -O annotations.csv
echo "download images.zip"
wget https://xxxxx -O image.zip
unzip images.zip
rm images.zip 

