#!/usr/bin/env python3
"""
Enhanced Bittensor Mining Script
================================

This script provides comprehensive functionality for:
1. Understanding Bittensor architecture and technicalities
2. Creating/importing wallets with proper security
3. Registering on subnets (specifically Synth subnet 50)
4. Implementing custom mining logic for synthetic data generation
5. Monitoring performance and rewards

Technical Architecture Understanding:
- Subtensor: The blockchain layer that coordinates all operations
- Subnets: Specialized AI service networks (each with unique requirements)
- Miners: Service providers that respond to validator requests
- Validators: Quality assessors that score miner responses
- Subnet Owners: Define incentive mechanisms and evaluation criteria

Usage Examples:
    python bittensor_mining_script.py --action create_wallet --wallet_name my_miner
    python bittensor_mining_script.py --action start_mining --wallet_name my_miner --subnet 50
    python bittensor_mining_script.py --action check_performance --wallet_name my_miner --subnet 50
"""

import argparse
import logging
import sys
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

import bittensor as bt
import numpy as np

# Configure simple logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SubnetInfo:
    """Data class for subnet information - SDK only"""
    uid: int
    burn_cost: float
    tempo: int
    max_validators: int
    difficulty: int
    immunity_period: int

class BittensorMiner:
    """Enhanced Bittensor Mining Client with comprehensive functionality"""
    
    def __init__(self, network: str = "test"):
        """
        Initialize the enhanced Bittensor miner
        
        Args:
            network: Network to connect to ("test" or "finney")
        """
        self.network = network
        self.subtensor = bt.subtensor(network=network)
        self.wallet = None
        self.axon = None  # Real Bittensor axon for mining
        self.mining_stats = {
            'requests_processed': 0,
            'start_time': None,
            'last_request_time': None
        }
        
    def create_wallet(self, wallet_name: str, hotkey_name: Optional[str] = None, 
                     password: Optional[str] = None) -> bt.wallet:
        """
        Create a new wallet or load existing one with enhanced security
        
        Args:
            wallet_name: Name of the coldkey wallet
            hotkey_name: Name of the hotkey (optional)
            password: Password for encrypted wallet (optional)
            
        Returns:
            Bittensor wallet object
        """
        try:
            logger.info("🔐 CREATING SECURE WALLET")
            logger.info("=" * 30)
            
            # Create wallet with hotkey if specified
            if hotkey_name:
                self.wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
                logger.info(f"Creating wallet with coldkey: {wallet_name}")
                logger.info(f"Creating wallet with hotkey: {hotkey_name}")
            else:
                self.wallet = bt.wallet(name=wallet_name)
                logger.info(f"Creating wallet: {wallet_name}")
            
            # Create wallet if it doesn't exist
            self.wallet.create_if_non_existent()
            
            logger.info("✅ Wallet created/loaded successfully!")
            
            # Display comprehensive wallet info
            self._display_detailed_wallet_info()
            
            # Security recommendations
            self._display_security_recommendations()
            
            return self.wallet
            
        except Exception as e:
            logger.error(f"❌ Failed to create wallet: {e}")
            raise
    
    def _display_detailed_wallet_info(self):
        """Display comprehensive wallet information"""
        if not self.wallet:
            return
            
        logger.info("\n📋 DETAILED WALLET INFORMATION")
        logger.info("=" * 35)
        logger.info(f"   Coldkey Address: {self.wallet.coldkey.ss58_address}")
        if self.wallet.hotkey:
            logger.info(f"   Hotkey Address: {self.wallet.hotkey.ss58_address}")
        
        # Check balance
        try:
            balance = self.subtensor.get_balance(self.wallet.coldkey.ss58_address)
            logger.info(f"   Balance: {balance.tao:.6f} τ")
            
            # Convert to USD (approximate)
            usd_value = balance.tao * 400  # Rough estimate
            logger.info(f"   Approximate USD Value: ${usd_value:.2f}")
            
        except Exception as e:
            logger.warning(f"   Could not fetch balance: {e}")
    
    def _display_security_recommendations(self):
        """Display security recommendations"""
        logger.info("\n🛡️  SECURITY RECOMMENDATIONS")
        logger.info("=" * 30)
        logger.info("1. Keep your mnemonic phrase secure and offline")
        logger.info("2. Use strong passwords for encrypted wallets")
        logger.info("3. Never share private keys or mnemonic phrases")
        logger.info("4. Consider using hardware wallets for large amounts")
        logger.info("5. Regularly backup your wallet files")
        logger.info("6. Only use official Bittensor tools")
    
    def get_subnet_information(self, subnet_uid: int) -> SubnetInfo:
        """
        Get subnet information from SDK only
        
        Args:
            subnet_uid: Subnet UID to check
            
        Returns:
            SubnetInfo object with SDK data only
        """
        try:
            subnet_info = self.subtensor.get_subnet_info(subnet_uid)
            
            return SubnetInfo(
                uid=subnet_uid,
                burn_cost=subnet_info.burn.tao,
                tempo=subnet_info.tempo,
                max_validators=subnet_info.max_allowed_validators,
                difficulty=subnet_info.difficulty,
                immunity_period=subnet_info.immunity_period
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to get subnet info: {e}")
            raise
    
    def check_subnet_info(self, subnet_uid: int):
        """
        Display technical subnet information from SDK
        
        Args:
            subnet_uid: Subnet UID to check
        """
        try:
            subnet_info = self.get_subnet_information(subnet_uid)
            
            logger.info(f"\n📊 SUBNET {subnet_uid} TECHNICAL INFORMATION")
            logger.info("=" * 50)
            logger.info(f"   UID: {subnet_info.uid}")
            logger.info(f"   Registration Cost: {subnet_info.burn_cost:.6f} τ")
            logger.info(f"   Tempo: {subnet_info.tempo}")
            logger.info(f"   Max Validators: {subnet_info.max_validators}")
            logger.info(f"   Difficulty: {subnet_info.difficulty}")
            logger.info(f"   Immunity Period: {subnet_info.immunity_period}")
            
            # Check if wallet is registered
            if self.wallet:
                is_registered = self.subtensor.is_hotkey_registered_on_subnet(
                    hotkey_ss58=self.wallet.hotkey.ss58_address,
                    netuid=subnet_uid
                )
                status = "✅ Registered" if is_registered else "❌ Not Registered"
                logger.info(f"\n   Registration Status: {status}")
            
        except Exception as e:
            logger.error(f"❌ Failed to get subnet info: {e}")
    
    def register_on_subnet(self, subnet_uid: int, wallet_name: str) -> bool:
        """
        Register on a specific subnet with enhanced error handling
        
        Args:
            subnet_uid: Subnet UID to register on
            wallet_name: Wallet name to use
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if not self.wallet:
                self.wallet = bt.wallet(name=wallet_name)
            
            logger.info(f"\n🔧 REGISTERING ON SUBNET {subnet_uid}")
            logger.info("=" * 35)
            
            # Get subnet info first
            subnet_info = self.get_subnet_information(subnet_uid)
            logger.info(f"   Subnet UID: {subnet_info.uid}")
            logger.info(f"   Cost: {subnet_info.burn_cost:.6f} τ")
            logger.info(f"   Tempo: {subnet_info.tempo}")
            
            # Check current balance
            balance = self.subtensor.get_balance(self.wallet.coldkey.ss58_address)
            logger.info(f"   Your Balance: {balance.tao:.6f} τ")
            
            if balance.tao < subnet_info.burn_cost:
                logger.error(f"❌ Insufficient balance! Need {subnet_info.burn_cost:.6f} τ, have {balance.tao:.6f} τ")
                logger.info("💡 Get testnet TAO from: https://test.taostats.io/")
                return False
            
            # Check if already registered
            if self.subtensor.is_hotkey_registered_on_subnet(
                hotkey_ss58=self.wallet.hotkey.ss58_address,
                netuid=subnet_uid
            ):
                logger.info(f"✅ Already registered on subnet {subnet_uid}")
                return True
            
            logger.info("   Proceeding with registration...")
            
            # Register on subnet using current API
            success = self.subtensor.register(
                wallet=self.wallet,
                netuid=subnet_uid,
                prompt=False
            )
            
            if success:
                logger.info(f"✅ Successfully registered on subnet {subnet_uid}!")
                logger.info("   You can now start mining on this subnet")
                return True
            else:
                logger.error(f"❌ Failed to register on subnet {subnet_uid}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Registration failed: {e}")
            return False
    
    def generate_synthetic_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate synthetic data for Synth subnet (50)
        
        Args:
            request_data: Request data from validator
            
        Returns:
            Generated synthetic data
        """
        try:
            # Extract request parameters
            data_type = request_data.get('data_type', 'text')
            size = request_data.get('size', 100)
            seed = request_data.get('seed', int(time.time()))
            
            logger.info(f"🔄 Generating synthetic {data_type} data (size: {size})")
            
            # Set random seed for reproducibility
            np.random.seed(seed)
            
            if data_type == 'text':
                # Generate synthetic text data
                synthetic_data = self._generate_synthetic_text(size)
            elif data_type == 'numerical':
                # Generate synthetic numerical data
                synthetic_data = self._generate_synthetic_numerical(size)
            elif data_type == 'categorical':
                # Generate synthetic categorical data
                synthetic_data = self._generate_synthetic_categorical(size)
            else:
                # Default to text
                synthetic_data = self._generate_synthetic_text(size)
            
            return {
                'data': synthetic_data,
                'metadata': {
                    'data_type': data_type,
                    'size': len(synthetic_data),
                    'seed': seed,
                    'generated_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate synthetic data: {e}")
            return {'error': str(e)}
    
    def _generate_synthetic_text(self, size: int) -> List[str]:
        """Generate synthetic text data"""
        # Sample text patterns for synthetic data
        patterns = [
            "The quick brown fox jumps over the lazy dog",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit",
            "Machine learning algorithms process data efficiently",
            "Artificial intelligence transforms industries worldwide",
            "Data science enables informed decision making"
        ]
        
        synthetic_texts = []
        for i in range(size):
            # Create variations of patterns
            pattern = patterns[i % len(patterns)]
            variation = f"{pattern} - Sample {i+1}"
            synthetic_texts.append(variation)
        
        return synthetic_texts
    
    def _generate_synthetic_numerical(self, size: int) -> List[float]:
        """Generate synthetic numerical data"""
        # Generate normally distributed data
        return np.random.normal(0, 1, size).tolist()
    
    def _generate_synthetic_categorical(self, size: int) -> List[str]:
        """Generate synthetic categorical data"""
        categories = ['A', 'B', 'C', 'D', 'E']
        return np.random.choice(categories, size).tolist()
    
    def start_mining(self, subnet_uid: int, wallet_name: str):
        """
        Start mining on a subnet with custom logic for Synth subnet
        
        Args:
            subnet_uid: Subnet UID to mine on
            wallet_name: Wallet name to use
        """
        try:
            if not self.wallet:
                self.wallet = bt.wallet(name=wallet_name)
            
            logger.info(f"\n⛏️  STARTING MINING ON SUBNET {subnet_uid}")
            logger.info("=" * 40)
            
            # Check if registered
            if not self.subtensor.is_hotkey_registered_on_subnet(
                hotkey_ss58=self.wallet.hotkey.ss58_address,
                netuid=subnet_uid
            ):
                logger.error(f"❌ Not registered on subnet {subnet_uid}")
                logger.info("💡 Register first using: --action register")
                return
            
            logger.info("✅ Registration verified")
            logger.info("🚀 Starting mining operations...")
            
            # Initialize mining stats
            self.mining_stats['start_time'] = datetime.now()
            self.mining_stats['requests_processed'] = 0
            
            # Create custom miner for Synth subnet
            # Synth subnet is 50 on mainnet, 247 on testnet
            synth_subnet_uid = 247 if self.network == "test" else 50
            if subnet_uid == synth_subnet_uid:  # Synth subnet
                self._start_synth_mining()
            else:
                # Generic mining for other subnets
                self._start_generic_mining(subnet_uid)
            
        except Exception as e:
            logger.error(f"❌ Mining failed: {e}")
            raise
    
    def _start_synth_mining(self):
        """Start mining specifically for Synth subnet using real Bittensor network"""
        logger.info("🎯 Starting REAL Synth subnet mining (synthetic data generation)")
        logger.info("   Connecting to actual Bittensor network...")
        logger.info("   Miners respond to validator requests with synthetic data")
        logger.info("   Rewards based on data quality and response speed")
        
        try:
            # Create axon for real mining
            self.axon = bt.axon(
                wallet=self.wallet,
                port=8091,  # Default axon port
                external_ip="0.0.0.0"  # Listen on all interfaces
            )
            
            # Define the forward function that responds to validator requests
            def forward_synth_data(synapse: bt.synapse) -> bt.synapse:
                """Real forward function that responds to validator requests"""
                try:
                    logger.info(f"📥 Received request from validator: {synapse.dendrite.hotkey}")
                    
                    # Extract request data from synapse
                    request_data = {
                        'data_type': getattr(synapse, 'data_type', 'text'),
                        'size': getattr(synapse, 'size', 100),
                        'seed': getattr(synapse, 'seed', int(time.time()))
                    }
                    
                    # Generate synthetic data
                    response = self.generate_synthetic_data(request_data)
                    
                    # Update stats
                    self.mining_stats['requests_processed'] += 1
                    self.mining_stats['last_request_time'] = datetime.now()
                    
                    logger.info(f"📊 Processed REAL request #{self.mining_stats['requests_processed']}")
                    logger.info(f"   Generated {response['metadata']['data_type']} data")
                    logger.info(f"   Size: {response['metadata']['size']} items")
                    
                    # Set response data in synapse
                    synapse.data = response['data']
                    synapse.metadata = response['metadata']
                    
                    return synapse
                    
                except Exception as e:
                    logger.error(f"❌ Error processing request: {e}")
                    synapse.data = []
                    synapse.metadata = {'error': str(e)}
                    return synapse
            
            # Attach forward function to axon
            self.axon.attach(forward_synth_data)
            
            # Start the axon (this connects to the network)
            logger.info("🚀 Starting axon and connecting to Bittensor network...")
            self.axon.start()
            
            logger.info("✅ Axon started successfully!")
            logger.info("   Waiting for validator requests...")
            logger.info("   Press Ctrl+C to stop mining")
            
            # Keep the axon running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Mining stopped by user")
            if self.axon:
                self.axon.stop()
            self._display_mining_summary()
        except Exception as e:
            logger.error(f"❌ Error in real mining: {e}")
            if self.axon:
                self.axon.stop()
            raise
    
    def _start_generic_mining(self, subnet_uid: int):
        """Start generic mining for other subnets using real Bittensor network"""
        logger.info(f"🎯 Starting REAL generic mining for subnet {subnet_uid}")
        logger.info("   Connecting to actual Bittensor network...")
        logger.info("   This is a generic implementation - implement subnet-specific logic for optimal performance")
        
        try:
            # Create axon for real mining
            self.axon = bt.axon(
                wallet=self.wallet,
                port=8091,  # Default axon port
                external_ip="0.0.0.0"  # Listen on all interfaces
            )
            
            # Define a generic forward function
            def forward_generic(synapse: bt.synapse) -> bt.synapse:
                """Generic forward function for any subnet"""
                try:
                    logger.info(f"📥 Received request from validator: {synapse.dendrite.hotkey}")
                    
                    # Update stats
                    self.mining_stats['requests_processed'] += 1
                    self.mining_stats['last_request_time'] = datetime.now()
                    
                    logger.info(f"📊 Processed REAL request #{self.mining_stats['requests_processed']}")
                    logger.info(f"   Subnet: {subnet_uid}")
                    logger.info(f"   Request type: Generic")
                    
                    # For generic subnets, we just acknowledge the request
                    # Subnet-specific logic should be implemented here
                    synapse.data = {"status": "processed", "subnet": subnet_uid}
                    synapse.metadata = {
                        "processed_at": datetime.now().isoformat(),
                        "subnet_uid": subnet_uid,
                        "request_count": self.mining_stats['requests_processed']
                    }
                    
                    return synapse
                    
                except Exception as e:
                    logger.error(f"❌ Error processing request: {e}")
                    synapse.data = {"error": str(e)}
                    synapse.metadata = {"error": str(e)}
                    return synapse
            
            # Attach forward function to axon
            self.axon.attach(forward_generic)
            
            # Start the axon (this connects to the network)
            logger.info("🚀 Starting axon and connecting to Bittensor network...")
            self.axon.start()
            
            logger.info("✅ Axon started successfully!")
            logger.info("   Waiting for validator requests...")
            logger.info("   Press Ctrl+C to stop mining")
            
            # Keep the axon running
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Mining stopped by user")
            if self.axon:
                self.axon.stop()
            self._display_mining_summary()
        except Exception as e:
            logger.error(f"❌ Error in real mining: {e}")
            if self.axon:
                self.axon.stop()
            raise
    
    def _display_mining_summary(self):
        """Display mining performance summary"""
        if self.mining_stats['start_time']:
            duration = datetime.now() - self.mining_stats['start_time']
            logger.info("\n📈 MINING PERFORMANCE SUMMARY")
            logger.info("=" * 35)
            logger.info(f"   Duration: {duration}")
            logger.info(f"   Requests Processed: {self.mining_stats['requests_processed']}")
            logger.info(f"   Requests Processed: {self.mining_stats['requests_processed']}")
            logger.info(f"   Last Request: {self.mining_stats['last_request_time']}")
            logger.info("   Note: Real rewards are distributed by the blockchain, not tracked locally")
    
    def check_performance(self, wallet_name: str, subnet_uid: Optional[int] = None):
        """
        Check mining performance and statistics
        
        Args:
            wallet_name: Wallet name to check
            subnet_uid: Specific subnet to check (optional)
        """
        try:
            if not self.wallet:
                self.wallet = bt.wallet(name=wallet_name)
            
            logger.info(f"\n📊 PERFORMANCE CHECK FOR WALLET: {wallet_name}")
            logger.info("=" * 45)
            
            # Check balance
            balance = self.subtensor.get_balance(self.wallet.coldkey.ss58_address)
            logger.info(f"   Current Balance: {balance.tao:.6f} τ")
            
            # Check registrations
            registrations = []
            subnets_to_check = [subnet_uid] if subnet_uid else range(256)
            
            for netuid in subnets_to_check:
                try:
                    if self.subtensor.is_hotkey_registered_on_subnet(
                        hotkey_ss58=self.wallet.hotkey.ss58_address,
                        netuid=netuid
                    ):
                        subnet_info = self.subtensor.get_subnet_info(netuid)
                        registrations.append((netuid, subnet_info.name))
                except:
                    continue
            
            if registrations:
                logger.info("\n   Active Registrations:")
                for netuid, name in registrations:
                    logger.info(f"   - Subnet {netuid}: {name}")
            else:
                logger.info("\n   No active registrations found")
            
            # Display mining stats if available
            if self.mining_stats['start_time']:
                self._display_mining_summary()
                
        except Exception as e:
            logger.error(f"❌ Failed to check performance: {e}")
    
    def list_my_registrations(self, wallet_name: str):
        """
        List all subnets where the wallet is registered with details
        
        Args:
            wallet_name: Wallet name to check
        """
        try:
            if not self.wallet:
                self.wallet = bt.wallet(name=wallet_name)
            
            logger.info(f"\n📋 REGISTRATIONS FOR WALLET: {wallet_name}")
            logger.info("=" * 40)
            
            registrations = []
            for netuid in range(256):  # Check first 256 subnets
                try:
                    if self.subtensor.is_hotkey_registered_on_subnet(
                        hotkey_ss58=self.wallet.hotkey.ss58_address,
                        netuid=netuid
                    ):
                        subnet_info = self.subtensor.get_subnet_info(netuid)
                        registrations.append((netuid, subnet_info.name))
                except:
                    continue
            
            if registrations:
                logger.info("   Active Registrations:")
                for netuid, name in registrations:
                    logger.info(f"   ✅ Subnet {netuid}: {name}")
            else:
                logger.info("   No registrations found")
                logger.info("   💡 Register on a subnet to start mining!")
                
        except Exception as e:
            logger.error(f"❌ Failed to list registrations: {e}")

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Bittensor Mining Script with Educational Features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new wallet
  python bittensor_mining_script.py --action create_wallet --wallet_name my_miner

  # Check subnet 247 (Synth) information
  python bittensor_mining_script.py --action check_subnet --subnet 247

  # Register on Synth subnet
  python bittensor_mining_script.py --action register --wallet_name my_miner --subnet 247

  # Start mining on Synth subnet
  python bittensor_mining_script.py --action start_mining --wallet_name my_miner --subnet 247

  # Check performance
  python bittensor_mining_script.py --action check_performance --wallet_name my_miner
        """
    )
    
    parser.add_argument("--action", required=True, 
                       choices=["create_wallet", "register", "start_mining", "check_subnet", 
                               "list_registrations", "check_performance"],
                       help="Action to perform")
    parser.add_argument("--wallet_name", help="Wallet name (required for most actions)")
    parser.add_argument("--hotkey_name", help="Hotkey name (optional)")
    parser.add_argument("--subnet", type=int, help="Subnet UID")
    parser.add_argument("--network", default="test", choices=["test", "finney"], 
                       help="Network to use (default: test)")
    parser.add_argument("--password", help="Password for encrypted wallet (optional)")
    
    # Parse only known arguments to avoid conflicts with Bittensor
    args, unknown = parser.parse_known_args()
    
    # Initialize enhanced miner
    miner = BittensorMiner(network=args.network)
    
    try:
        if args.action == "create_wallet":
            if not args.wallet_name:
                logger.error("❌ Wallet name required for create_wallet")
                sys.exit(1)
            miner.create_wallet(args.wallet_name, args.hotkey_name, args.password)
            
        elif args.action == "register":
            if not args.wallet_name:
                logger.error("❌ Wallet name required for registration")
                sys.exit(1)
            if not args.subnet:
                logger.error("❌ Subnet UID required for registration")
                sys.exit(1)
            miner.register_on_subnet(args.subnet, args.wallet_name)
            
        elif args.action == "start_mining":
            if not args.wallet_name:
                logger.error("❌ Wallet name required for mining")
                sys.exit(1)
            if not args.subnet:
                logger.error("❌ Subnet UID required for mining")
                sys.exit(1)
            miner.start_mining(args.subnet, args.wallet_name)
            
        elif args.action == "check_subnet":
            if not args.subnet:
                logger.error("❌ Subnet UID required")
                sys.exit(1)
            miner.check_subnet_info(args.subnet)
            
        elif args.action == "list_registrations":
            if not args.wallet_name:
                logger.error("❌ Wallet name required for list_registrations")
                sys.exit(1)
            miner.list_my_registrations(args.wallet_name)
            
        elif args.action == "check_performance":
            if not args.wallet_name:
                logger.error("❌ Wallet name required for check_performance")
                sys.exit(1)
            miner.check_performance(args.wallet_name, args.subnet)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Operation cancelled by user")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 