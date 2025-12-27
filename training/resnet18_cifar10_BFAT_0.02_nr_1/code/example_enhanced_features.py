"""
Example: Using Enhanced Feature Extractor with RL Restorer

This demonstrates how to use the enhanced feature extractor that captures
more subtle differences in feature maps.
"""

import torch
from util.enhanced_feature_extractor import EnhancedSensitiveActivationCollector
from util.rl_restorer import RLRestorer


def example_enhanced_features():
    """Example of using enhanced features"""
    
    # Setup
    batch_size = 32
    num_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create enhanced collector
    # collector = EnhancedSensitiveActivationCollector(model, sensitive_info, baseline_stats)
    
    # Example workflow:
    print("Enhanced Feature Extraction Workflow:")
    print("=" * 60)
    
    print("\n1. Collect Clean Activations:")
    print("   collector.clear()")
    print("   fault_injector.disable()")
    print("   with torch.no_grad():")
    print("       _ = model(inputs)")
    print("   clean_activations = collector.buffers.copy()")
    
    print("\n2. Collect Faulted Activations:")
    print("   collector.clear()")
    print("   fault_injector.enable()")
    print("   with torch.no_grad():")
    print("       _ = model(inputs)")
    print("   fault_injector.disable()")
    
    print("\n3. Extract Enhanced Features:")
    print("   features = collector.build_enhanced_feature_vector(")
    print("       device=device,")
    print("       clean_activations=clean_activations,  # Key: provide clean activations")
    print("   )")
    
    print("\n4. Use with RL Restorer:")
    print("   feature_dim = features.size(1)  # Much larger than before!")
    print("   restorer = RLRestorer(")
    print("       feature_dim=feature_dim,")
    print("       num_classes=num_classes,")
    print("       ...")
    print("   )")
    print("   logits_restored, info = restorer(logits_faulted, features, ...)")


def compare_feature_dims():
    """Compare feature dimensions between old and new methods"""
    
    print("\nFeature Dimension Comparison:")
    print("=" * 60)
    
    # Example: 3 layers, 64 channels each
    num_layers = 3
    num_channels = 64
    
    # Old method: 4 features per channel
    old_dim = 4 * num_channels * num_layers
    print(f"\nOld Method (4 features per channel):")
    print(f"  {num_layers} layers × {num_channels} channels × 4 features = {old_dim:,} features")
    
    # New method: ~78 features per channel (estimated)
    features_per_channel = {
        'basic_stats': 5,           # energy, mean, std, max, min
        'distribution': 7,          # 5 percentiles + skewness + kurtosis
        'spatial': 50,              # multi-scale grids + correlations
        'frequency': 5,             # low/high freq + distribution
        'relative': 6,              # differences + correlation
        'gradient': 5,              # gradient statistics
    }
    new_dim_per_channel = sum(features_per_channel.values())
    new_dim = new_dim_per_channel * num_channels * num_layers
    
    print(f"\nNew Method (Enhanced Features):")
    for name, count in features_per_channel.items():
        print(f"  {name}: {count} features")
    print(f"  Total per channel: {new_dim_per_channel} features")
    print(f"  {num_layers} layers × {num_channels} channels × {new_dim_per_channel} features = {new_dim:,} features")
    
    print(f"\nIncrease: {new_dim / old_dim:.1f}x more features")
    print(f"  Old: {old_dim:,} features")
    print(f"  New: {new_dim:,} features")
    print(f"  Difference: {new_dim - old_dim:,} features")


def example_selective_features():
    """Example of using selective features (balance performance and effect)"""
    
    print("\nSelective Feature Strategy:")
    print("=" * 60)
    
    print("\nOption 1: Essential Features Only")
    print("  - Basic stats: 5 features (energy, mean, std, max, min)")
    print("  - Distribution: 7 features (percentiles, skewness, kurtosis)")
    print("  - Relative: 6 features (differences, correlation) - MOST IMPORTANT!")
    print("  Total: ~18 features per channel")
    print("  Reason: Relative features directly capture fault-induced changes")
    
    print("\nOption 2: Progressive Feature Addition")
    print("  - Epoch 1-10: Basic stats only (5 features)")
    print("  - Epoch 11-20: Add distribution features (+7 features)")
    print("  - Epoch 21+: Add relative features (+6 features)")
    print("  Reason: Gradually increase complexity, help network learn")
    
    print("\nOption 3: Full Features (for high BER)")
    print("  - All features: ~78 features per channel")
    print("  Reason: Maximum information, best for difficult cases")


if __name__ == '__main__':
    example_enhanced_features()
    compare_feature_dims()
    example_selective_features()


