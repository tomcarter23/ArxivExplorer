sudo yum update -y && \
sudo yum install docker -y && \
sudo yum install git -y && \
sudo yum install libicu -y && \
sudo systemctl enable docker

# set swap on
sudo dd if=/dev/zero of=/swapfile bs=256M count=32
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon -s
echo "/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab
