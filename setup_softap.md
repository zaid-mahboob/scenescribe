sudo nano /etc/systemd/system/softap.service

"add the code below to softap.service"
[Unit]
Description=SoftAP Access Point Setup
After=network.target

[Service]
Type=oneshot
ExecStart=/home/scenescribe/Desktop/scenescribe/setup_softap.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target

chmod +x /path/to/your/softap_script.sh
<!-- if not executable -->

sudo systemctl daemon-reload
sudo systemctl enable softap.service

sudo systemctl start softap.service

# Check status
sudo systemctl status softap.service

# View logs
journalctl -u softap.service -b