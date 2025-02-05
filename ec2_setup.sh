sudo apt-get update -y && \
sudo apt-get install docker-compose -y && \
sudo apt-get install git -y && \
sudo apt-get install libicu -y && \
sudo systemctl enable docker

# set swap on
sudo dd if=/dev/zero of=/swapfile bs=256M count=32
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo swapon -s
echo "/swapfile swap swap defaults 0 0" | sudo tee -a /etc/fstab
