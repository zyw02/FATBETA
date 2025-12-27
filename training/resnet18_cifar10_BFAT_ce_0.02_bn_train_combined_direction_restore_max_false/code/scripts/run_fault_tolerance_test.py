#!/usr/bin/env python3
"""
Script to run fault tolerance test for ResNet18 SR-QAT model
Test different BER values: 1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1

Note: Currently, the highest bit protection per layer is not implemented.
This requires custom modification to the FaultInjector class.
"""

import os
import subprocess
import yaml
import argparse

def run_fault_tolerance_test(config_path: str, ber_values: list):
    """
    Run fault tolerance test with different BER values

    Args:
        config_path: Path to the evaluation config file
        ber_values: List of BER values to test
    """

    print("🚀 Starting Fault Tolerance Test for ResNet18 SR-QAT Model")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"BER values to test: {ber_values}")
    print()

    results = []

    for ber in ber_values:
        print(f"🔬 Testing BER = {ber}")
        print("-" * 40)

        # Modify config file to set current BER
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        config['fault_aware_training']['ber'] = ber

        # Create temporary config file
        temp_config_path = f"{config_path}.temp_ber_{ber}"
        with open(temp_config_path, 'w') as f:
            yaml.safe_dump(config, f)

        try:
            # Run evaluation - main_normal.py takes config file as positional argument
            cmd = f"python main_normal.py {temp_config_path}"
            print(f"Running: {cmd}")

            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

            accuracy_found = False
            accuracy = None

            if result.returncode == 0:
                print("✅ Test completed successfully")

                # Try to extract Top1 accuracy from different patterns
                lines = result.stdout.split('\n')

                # Look for the output format from main_normal.py eval mode
                # It prints a list like [32.46] which is the Top-1 accuracy
                import re
                list_pattern = r'\[([0-9.]+)\]'
                match = re.search(list_pattern, result.stdout.strip())

                if match:
                    accuracy = match.group(1)
                    print(f"📊 BER {ber}: Top-1 Accuracy = {accuracy}%")
                    accuracy_found = True
                else:
                    # Fallback to other patterns
                    patterns = [
                        r'Top1.*: ([0-9.]+)',
                        r'Top-1.*: ([0-9.]+)',
                        r'Top-1 Acc.* ([0-9.]+)',
                        r'top1.*: ([0-9.]+)',
                        r'Accuracy.*: ([0-9.]+)',
                        r'Acc.*: ([0-9.]+)%?'
                    ]

                    for line in reversed(lines):
                        for pattern in patterns:
                            match = re.search(pattern, line, re.IGNORECASE)
                            if match:
                                accuracy = match.group(1)
                                print(f"📊 BER {ber}: Top-1 Accuracy = {accuracy}%")
                                accuracy_found = True
                                break
                        if accuracy_found:
                            break

                if not accuracy_found:
                    print(f"⚠️ BER {ber}: Could not extract accuracy from output")
                    # Show last 20 lines for debugging
                    print("Last 20 lines of output:")
                    for line in lines[-20:]:
                        if line.strip():
                            print(f"  {line}")
            else:
                print("❌ Test failed")
                print("Error output:")
                print(result.stderr)

            results.append({
                'ber': ber,
                'returncode': result.returncode,
                'accuracy': accuracy if accuracy_found else None,
                'stdout': result.stdout,
                'stderr': result.stderr
            })

        except Exception as e:
            print(f"❌ Error running test for BER {ber}: {e}")
            results.append({
                'ber': ber,
                'error': str(e)
            })

        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

        print()

    # Print summary
    print("📊 Test Summary")
    print("=" * 60)
    for result in results:
        ber = result['ber']
        if 'error' in result:
            print(f"BER {ber}: ERROR - {result['error']}")
        elif result['returncode'] == 0:
            accuracy = result.get('accuracy')
            if accuracy:
                print(f"BER {ber}: Top-1 Accuracy = {accuracy}%")
            else:
                print(f"BER {ber}: SUCCESS (accuracy not extracted)")
        else:
            print(f"BER {ber}: FAILED (return code {result['returncode']})")

    return results

def main():
    parser = argparse.ArgumentParser(description='Run fault tolerance test')
    parser.add_argument('--config', type=str,
                       default='configs/eval/eval_resnet18_cifar10_fault_tolerance_test.yaml',
                       help='Path to evaluation config file')
    parser.add_argument('--ber-values', type=float, nargs='+',
                       default=[1e-4, 1e-3, 1e-2, 2e-2, 3e-2, 5e-2, 1e-1],
                       help='BER values to test')

    args = parser.parse_args()

    run_fault_tolerance_test(args.config, args.ber_values)

if __name__ == "__main__":
    main()
