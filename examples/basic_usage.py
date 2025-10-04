"""
Basic usage example for NanoAccel.
"""

from nanoaccel import NanoAccel, QuantizationConfig


def main():
    """Basic usage example."""
    print("NanoAccel Basic Usage Example")
    print("=" * 40)
    
    # Initialize NanoAccel with quantization
    print("1. Initializing NanoAccel with INT4 quantization...")
    quant_config = QuantizationConfig(
        enabled=True,
        quant_type="int4",
        compute_dtype=torch.float32
    )
    
    nanoaccel = NanoAccel(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        quant_config=quant_config,
        mixed_precision=True,
        verbose=True
    )
    
    # Load the model
    print("\n2. Loading model...")
    nanoaccel.load_model()
    
    # Generate text
    print("\n3. Generating text...")
    result = nanoaccel.generate(
        prompt="Write a short story about a robot learning to paint:",
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9
    )
    
    print(f"\nGenerated text:")
    print("-" * 50)
    print(result["text"])
    print("-" * 50)
    
    # Display performance statistics
    print(f"\nPerformance Statistics:")
    print(f"Tokens generated: {result['tokens_generated']}")
    print(f"Generation time: {result['generation_time']:.2f} seconds")
    print(f"Tokens per second: {result['tokens_per_second']:.2f}")
    
    # Get overall statistics
    stats = nanoaccel.get_stats()
    print(f"\nOverall Statistics:")
    print(f"Total tokens generated: {stats['total_tokens']}")
    print(f"Total time: {stats['total_time']:.2f} seconds")
    print(f"Average tokens/sec: {stats.get('average_tokens_per_second', 0):.2f}")


if __name__ == "__main__":
    import torch
    main()
