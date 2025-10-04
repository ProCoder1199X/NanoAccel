"""
Speculative decoding example for NanoAccel.
"""

from nanoaccel import NanoAccel, QuantizationConfig


def main():
    """Speculative decoding example."""
    print("NanoAccel Speculative Decoding Example")
    print("=" * 45)
    
    # Initialize NanoAccel with quantization and speculative decoding
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
    
    # Load the main model
    print("\n2. Loading main model...")
    nanoaccel.load_model()
    
    # Load draft model for speculative decoding
    print("\n3. Loading draft model for speculative decoding...")
    nanoaccel.load_draft_model("EleutherAI/pythia-70m")
    
    # Generate text with speculative decoding
    print("\n4. Generating text with speculative decoding...")
    result = nanoaccel.generate(
        prompt="Explain the concept of artificial intelligence in simple terms:",
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        use_speculative=True,
        gamma=4,
        early_exit_threshold=0.9
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
    print(f"Method: {result['method']}")
    
    # Display speculative decoding statistics
    if nanoaccel.speculative_decoder:
        spec_stats = nanoaccel.speculative_decoder.get_stats()
        print(f"\nSpeculative Decoding Statistics:")
        print(f"Acceptance rate: {spec_stats['acceptance_rate']:.2%}")
        print(f"Estimated speedup: {spec_stats['speedup_factor']:.2f}x")
        print(f"Draft tokens generated: {spec_stats['draft_tokens_generated']}")
        print(f"Accepted tokens: {spec_stats['accepted_tokens']}")
        print(f"Rejected tokens: {spec_stats['rejected_tokens']}")
    
    # Compare with standard generation
    print(f"\n5. Comparing with standard generation...")
    nanoaccel.reset_stats()
    
    standard_result = nanoaccel.generate(
        prompt="Explain the concept of artificial intelligence in simple terms:",
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        use_speculative=False
    )
    
    print(f"\nStandard Generation Performance:")
    print(f"Tokens generated: {standard_result['tokens_generated']}")
    print(f"Generation time: {standard_result['generation_time']:.2f} seconds")
    print(f"Tokens per second: {standard_result['tokens_per_second']:.2f}")
    
    # Calculate speedup
    if result['generation_time'] > 0 and standard_result['generation_time'] > 0:
        speedup = standard_result['generation_time'] / result['generation_time']
        print(f"\nActual Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    import torch
    main()
