
echo "starting donwload !\n"
cd SageMaker/mnt/efs
mkdir Lizard && cd Lizard
curl 'xxxxxx-kaggle-link' -L -o 'archive.zip'
unzip archive.zip
rm archive.zip
