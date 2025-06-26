#!/bin/bash

# Configuration
SSID="SceneScribe-b24b49"
PASSWORD="scenescribe.pi1234"  # 8+ chars!
CHANNEL="6"
AP_INTERFACE="wlan1"
AP_IP="192.168.4.1"
CLIENT_INTERFACE="wlan0"

# Install dependencies
apt update
apt install -y hostapd dnsmasq iptables iptables-persistent

# Stop services
systemctl stop hostapd dnsmasq 2>/dev/null
pkill hostapd dnsmasq 2>/dev/null

# Configure hostapd (WPA2 - more reliable)
cat > /etc/hostapd/hostapd.conf <<EOF
interface=${AP_INTERFACE}
driver=nl80211
ssid=${SSID}
hw_mode=g
channel=${CHANNEL}
wpa=2
wpa_passphrase=${PASSWORD}
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF

# Configure hostapd service
sed -i 's|^#DAEMON_CONF=""|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd

# Configure dnsmasq
cat > /etc/dnsmasq.conf <<EOF
interface=${AP_INTERFACE}
dhcp-range=192.168.4.2,192.168.4.254,24h
EOF

# Network setup
ip link set ${AP_INTERFACE} down
ip addr flush dev ${AP_INTERFACE}
ip link set ${AP_INTERFACE} up
ip addr add ${AP_IP}/24 dev ${AP_INTERFACE}

# Enable services
systemctl unmask hostapd
systemctl enable hostapd

# Start services with delays
systemctl start hostapd
sleep 2
systemctl start dnsmasq

# NAT rules
iptables -t nat -A POSTROUTING -o ${CLIENT_INTERFACE} -j MASQUERADE
iptables -A FORWARD -i ${AP_INTERFACE} -o ${CLIENT_INTERFACE} -j ACCEPT
sysctl -w net.ipv4.ip_forward=1
iptables-save > /etc/iptables/rules.v4

echo "SoftAP ${SSID} should be running on ${AP_IP}"
echo "Check status with: sudo systemctl status hostapd"