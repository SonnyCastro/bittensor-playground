# Bittensor Mining Script

## 1. Create and Activate a Virtual Environment

```bash
# Create a virtual environment (if you haven't already)
python3 -m venv bt_venv

# Activate the virtual environment (macOS/Linux)
source bt_venv/bin/activate
```

**You should see (bt_venv) at the start of your terminal prompt when it's active.**

---

## 🚀 Quick Start

### 1. Setup
```bash
# Install Bittensor (includes all other dependencies)
pip install bittensor>=6.0.0

# Or install from requirements (optional)
pip install -r requirements.txt
```

### 2. Create Wallet
```bash
python bittensor_mining_script.py --action create_wallet --wallet_name test_miner
```

### 3. Get Testnet TAO

### 4. Check Subnet Info
```bash
python bittensor_mining_script.py --action check_subnet --subnet 247
```

### 5. Register on Subnet
```bash
python bittensor_mining_script.py --action register --wallet_name test_miner --subnet 247
```

### 6. Start Mining
```bash
python bittensor_mining_script.py --action start_mining --wallet_name test_miner --subnet 247
```

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `create_wallet` | Create a new wallet |
| `check_subnet` | View subnet information |
| `register` | Register on a subnet |
| `start_mining` | Start mining operations |
| `list_registrations` | List your registrations |
| `check_performance` | Check mining performance |

## 🎯 Synth Subnet (247 on Testnet)

The Synth subnet focuses on synthetic data generation:
- **Purpose**: Generate high-quality synthetic datasets
- **Data Types**: Text, numerical, categorical data
- **Requirements**: Fast response times, diverse outputs
- **Rewards**: Based on data quality and response speed

## 🔧 Usage Examples

```bash
# Create wallet with hotkey
python bittensor_mining_script.py --action create_wallet --wallet_name test_miner --hotkey_name my_hotkey

# Check subnet 247 details
python bittensor_mining_script.py --action check_subnet --subnet 247

# Register on subnet 247
python bittensor_mining_script.py --action register --wallet_name test_miner --subnet 247

# Start mining on subnet 247
python bittensor_mining_script.py --action start_mining --wallet_name test_miner --subnet 247

# Check your performance
python bittensor_mining_script.py --action check_performance --wallet_name test_miner

# List all your registrations
python bittensor_mining_script.py --action list_registrations --wallet_name test_miner
```

## 🛡️ Security

- **Keep your mnemonic phrase secure and offline**
- **Use strong passwords for encrypted wallets**
- **Never share private keys or mnemonic phrases**
- **Only use official Bittensor tools**

## 📊 Performance Monitoring

The script tracks:
- Requests processed
- Total rewards earned
- Response times
- Error rates

## 🔗 Resources

- [Bittensor Documentation](https://docs.bittensor.org)
- [Bittensor Discord](https://discord.gg/bittensor)

---
