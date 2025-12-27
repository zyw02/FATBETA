"""
Example: Integrating RL Restorer with Enhanced Features into Training Pipeline

This shows how to integrate RL Restorer with enhanced feature extraction
into the existing training code.
"""

import torch
import torch.nn.functional as F
from util.rl_restorer_integration import (
    create_rl_restorer_with_features,
    extract_features_for_rl_restorer,
)
from util.rl_restorer_trainer import RLRestorerTrainer


def example_integration_into_training():
    """
    Example of integrating RL Restorer with enhanced features into training
    """
    
    print("=" * 60)
    print("RL Restorer Integration Example")
    print("=" * 60)
    
    print("\n1. Create RL Restorer with Feature Collector:")
    print("-" * 60)
    print("""
    from util.rl_restorer_integration import create_rl_restorer_with_features
    
    # Option A: Use enhanced features (recommended for high BER)
    restorer, collector = create_rl_restorer_with_features(
        model=model,
        sensitive_info=sensitive_info,
        baseline_stats=baseline_stats,
        num_classes=10,
        use_enhanced_features=True,  # Use enhanced features
        state_dim=128,
        hidden_dim=256,
        max_steps=3,
        device=device,
    )
    
    # Option B: Use standard features (faster, less expressive)
    restorer, collector = create_rl_restorer_with_features(
        model=model,
        sensitive_info=sensitive_info,
        baseline_stats=baseline_stats,
        num_classes=10,
        use_enhanced_features=False,  # Use standard features
        state_dim=128,
        hidden_dim=256,
        max_steps=3,
        device=device,
    )
    """)
    
    print("\n2. Create Trainer:")
    print("-" * 60)
    print("""
    from util.rl_restorer_trainer import RLRestorerTrainer
    
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
    """)
    
    print("\n3. Training Loop:")
    print("-" * 60)
    print("""
    from util.rl_restorer_integration import extract_features_for_rl_restorer
    
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get clean logits (for reward computation)
            fault_injector.disable()
            with torch.no_grad():
                logits_clean = model(inputs)
            
            # Get faulted logits
            fault_injector.enable()
            with torch.no_grad():
                logits_faulted = model(inputs)
            fault_injector.disable()
            
            # Extract features (automatically handles enhanced vs standard)
            features, _ = extract_features_for_rl_restorer(
                collector=collector,
                model=model,
                inputs=inputs,
                fault_injector=fault_injector,
                use_enhanced_features=restorer.use_enhanced_features,
                device=device,
            )
            
            if features is None:
                continue
            
            # Train RL Restorer
            metrics = trainer.train_step(
                logits_faulted=logits_faulted,
                features=features,
                logits_clean=logits_clean,
            )
            
            # Print metrics
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}:")
                print(f"  Actor Loss: {metrics['actor_loss']:.4f}")
                print(f"  Critic Loss: {metrics['critic_loss']:.4f}")
                print(f"  Mean Reward: {metrics['mean_reward']:.4f}")
        
        # Update learning rate
        trainer.update_schedulers()
    """)
    
    print("\n4. Inference:")
    print("-" * 60)
    print("""
    restorer.eval()
    with torch.no_grad():
        # Get faulted logits
        fault_injector.enable()
        logits_faulted = model(inputs)
        fault_injector.disable()
        
        # Extract features
        features, _ = extract_features_for_rl_restorer(
            collector=collector,
            model=model,
            inputs=inputs,
            fault_injector=fault_injector,
            use_enhanced_features=restorer.use_enhanced_features,
            device=device,
        )
        
        # Restore
        logits_restored, info = restorer(
            logits_faulted=logits_faulted,
            features=features,
            training=False,
            deterministic=True,
        )
        
        # Predict
        pred_restored = logits_restored.argmax(dim=1)
    """)


def example_feature_comparison():
    """Compare standard vs enhanced features"""
    
    print("\n" + "=" * 60)
    print("Feature Comparison")
    print("=" * 60)
    
    print("\nStandard Features (4 per channel):")
    print("  - Energy")
    print("  - Mean")
    print("  - Std")
    print("  - Max")
    print("  → Simple, fast, but limited information")
    
    print("\nEnhanced Features (~78 per channel):")
    print("  - Basic stats (5): energy, mean, std, max, min")
    print("  - Distribution (7): percentiles, skewness, kurtosis")
    print("  - Spatial (~50): multi-scale grids, correlations")
    print("  - Frequency (5): FFT-based features")
    print("  - Relative (6): differences from clean - MOST IMPORTANT!")
    print("  - Gradient (5): local changes, edges")
    print("  → Rich, expressive, captures subtle differences")
    
    print("\nWhen to use Enhanced Features:")
    print("  ✅ High BER scenarios (BER > 5e-2)")
    print("  ✅ When standard features don't work")
    print("  ✅ When you have compute resources")
    print("  ✅ When fault patterns are subtle")
    
    print("\nWhen to use Standard Features:")
    print("  ✅ Low BER scenarios (BER < 2e-2)")
    print("  ✅ When compute resources are limited")
    print("  ✅ For quick prototyping")
    print("  ✅ When standard features work well")


def example_progressive_feature_usage():
    """Example of progressive feature usage during training"""
    
    print("\n" + "=" * 60)
    print("Progressive Feature Usage Strategy")
    print("=" * 60)
    
    print("""
    # Strategy: Start with standard features, switch to enhanced later
    
    # Phase 1: Epochs 1-10 - Use standard features
    restorer_standard, collector_standard = create_rl_restorer_with_features(
        model, sensitive_info, baseline_stats, num_classes=10,
        use_enhanced_features=False,
    )
    trainer_standard = RLRestorerTrainer(restorer_standard, ...)
    
    # Train with standard features
    for epoch in range(10):
        # ... training loop with standard features ...
        pass
    
    # Phase 2: Epochs 11+ - Switch to enhanced features
    # Option A: Create new restorer with enhanced features
    restorer_enhanced, collector_enhanced = create_rl_restorer_with_features(
        model, sensitive_info, baseline_stats, num_classes=10,
        use_enhanced_features=True,
    )
    
    # Option B: Transfer learning (copy actor/critic weights, retrain state encoder)
    # restorer_enhanced.state_encoder = ...  # New encoder for enhanced features
    # restorer_enhanced.actor.load_state_dict(restorer_standard.actor.state_dict())
    # restorer_enhanced.critic.load_state_dict(restorer_standard.critic.state_dict())
    
    trainer_enhanced = RLRestorerTrainer(restorer_enhanced, ...)
    
    # Train with enhanced features
    for epoch in range(10, num_epochs):
        # ... training loop with enhanced features ...
        pass
    """)


if __name__ == '__main__':
    example_integration_into_training()
    example_feature_comparison()
    example_progressive_feature_usage()


