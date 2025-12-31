"""
Example: How to use RL Restorer

This script demonstrates how to integrate RL Restorer into your training pipeline.
"""

import torch
import torch.nn.functional as F
from util.rl_restorer import RLRestorer
from util.rl_restorer_trainer import RLRestorerTrainer


def example_basic_usage():
    """Basic usage example"""
    # Setup
    batch_size = 32
    num_classes = 10
    feature_dim = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create RL Restorer
    restorer = RLRestorer(
        feature_dim=feature_dim,
        num_classes=num_classes,
        state_dim=128,
        hidden_dim=256,
        max_steps=3,
        gamma=0.99,
        use_statistical_preprocessing=True,
    ).to(device)
    
    # Create trainer
    trainer = RLRestorerTrainer(
        restorer=restorer,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=1.0,
        use_ppo=True,
    )
    
    # Dummy data
    logits_faulted = torch.randn(batch_size, num_classes, device=device)
    logits_clean = torch.randn(batch_size, num_classes, device=device)
    features = torch.randn(batch_size, feature_dim, device=device)
    
    # Update statistics (optional, for statistical preprocessing)
    restorer.update_statistics(logits_clean)
    
    # Training step
    restorer.train()
    metrics = trainer.train_step(
        logits_faulted=logits_faulted,
        features=features,
        logits_clean=logits_clean,
    )
    
    print("Training Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Inference
    restorer.eval()
    with torch.no_grad():
        logits_restored, info = restorer(
            logits_faulted=logits_faulted,
            features=features,
            training=False,
            deterministic=True,
        )
    
    print(f"\nInference:")
    print(f"  Original logits shape: {logits_faulted.shape}")
    print(f"  Restored logits shape: {logits_restored.shape}")
    print(f"  Final reward: {info.get('final_reward', 0.0):.4f}")


def example_integration_with_training():
    """Example of integrating RL Restorer into existing training loop"""
    
    # This is a template - adapt to your actual training code
    def train_with_rl_restorer(model, train_loader, fault_injector, collector, device, num_epochs=10):
        # Setup
        feature_dim = 512  # Adjust based on your feature extractor
        num_classes = 10   # Adjust based on your dataset
        
        # Create RL Restorer
        restorer = RLRestorer(
            feature_dim=feature_dim,
            num_classes=num_classes,
            state_dim=128,
            hidden_dim=256,
            max_steps=3,
            use_statistical_preprocessing=True,
        ).to(device)
        
        # Create trainer
        trainer = RLRestorerTrainer(
            restorer=restorer,
            actor_lr=3e-4,
            critic_lr=3e-4,
            max_grad_norm=1.0,
            use_ppo=True,
        )
        
        # Collect statistics for preprocessing (optional)
        print("Collecting statistics for preprocessing...")
        logits_clean_list = []
        model.eval()
        with torch.no_grad():
            for inputs, _ in train_loader:
                inputs = inputs.to(device)
                fault_injector.disable()
                logits_clean = model(inputs)
                logits_clean_list.append(logits_clean)
                if len(logits_clean_list) * inputs.size(0) > 1000:  # Collect 1000 samples
                    break
        
        logits_clean_all = torch.cat(logits_clean_list, dim=0)
        restorer.update_statistics(logits_clean_all)
        print("Statistics collected!")
        
        # Training loop
        for epoch in range(num_epochs):
            model.eval()  # Model should be in eval mode for restorer training
            restorer.train()
            
            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_reward = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Get clean logits
                fault_injector.disable()
                with torch.no_grad():
                    logits_clean = model(inputs)
                
                # Get faulted logits
                fault_injector.enable()
                with torch.no_grad():
                    logits_faulted = model(inputs)
                fault_injector.disable()
                
                # Get features
                collector.clear()
                fault_injector.enable()
                with torch.no_grad():
                    _ = model(inputs)  # Forward to collect features
                fault_injector.disable()
                
                features = collector.build_feature_vector(device)
                if features is None:
                    continue
                
                # Train restorer
                metrics = trainer.train_step(
                    logits_faulted=logits_faulted,
                    features=features,
                    logits_clean=logits_clean,
                )
                
                total_actor_loss += metrics['actor_loss']
                total_critic_loss += metrics['critic_loss']
                total_reward += metrics['mean_reward']
                
                # Print progress
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}:")
                    print(f"  Actor Loss: {metrics['actor_loss']:.4f}")
                    print(f"  Critic Loss: {metrics['critic_loss']:.4f}")
                    print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
                    print(f"  Entropy: {metrics['entropy']:.4f}")
            
            # Update learning rate
            trainer.update_schedulers()
            
            # Epoch summary
            num_batches = len(train_loader)
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Avg Actor Loss: {total_actor_loss / num_batches:.4f}")
            print(f"  Avg Critic Loss: {total_critic_loss / num_batches:.4f}")
            print(f"  Avg Reward: {total_reward / num_batches:.4f}")
            print(f"  Learning Rates: {trainer.get_lr()}")
        
        return restorer, trainer


def example_inference_only():
    """Example of using RL Restorer for inference only"""
    
    # Load trained restorer
    # restorer = torch.load('rl_restorer.pth')
    
    # Or create a new one
    restorer = RLRestorer(
        feature_dim=512,
        num_classes=10,
        state_dim=128,
        hidden_dim=256,
        max_steps=3,
    )
    
    restorer.eval()
    
    # During inference
    def restore_logits(logits_faulted, features, device):
        restorer.eval()
        with torch.no_grad():
            logits_restored, info = restorer(
                logits_faulted=logits_faulted.to(device),
                features=features.to(device),
                training=False,
                deterministic=True,  # Use deterministic policy
            )
        return logits_restored, info
    
    # Example usage
    batch_size = 32
    num_classes = 10
    feature_dim = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logits_faulted = torch.randn(batch_size, num_classes)
    features = torch.randn(batch_size, feature_dim)
    
    logits_restored, info = restore_logits(logits_faulted, features, device)
    
    print(f"Restored logits shape: {logits_restored.shape}")
    print(f"Number of steps used: {info.get('num_steps', 'N/A')}")


if __name__ == '__main__':
    print("=" * 60)
    print("RL Restorer Example")
    print("=" * 60)
    
    print("\n1. Basic Usage:")
    print("-" * 60)
    example_basic_usage()
    
    print("\n2. Integration Example (template):")
    print("-" * 60)
    print("See example_integration_with_training() function")
    
    print("\n3. Inference Only Example:")
    print("-" * 60)
    example_inference_only()


