sudo yum update -y
sudo amazon-linux-extras install epel -y
sudo yum install openslide-tools -y

# install GDC client to download from the TCGA Repository
pip install virtualenv 
git clone https://github.com/NCI-GDC/gdc-client
cd bin 
chmod +x package
./package