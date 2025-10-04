"""
Command-line interface for NanoAccel.
"""

import argparse
import sys
import json
from typing import Dict, Any
import logging

from .core import NanoAccel
from .quantization import QuantizationConfig
from .utils import check_system_requirements, detect_cpu_features


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="NanoAccel: CPU-Optimized LLM Accelerator for Low-End Hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  nanoaccel --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello, world!"
  
  # With quantization
  nanoaccel --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --quant int4 --prompt "Tell me a story"
  
  # With speculative decoding
  nanoaccel --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --speculative --draft-model EleutherAI/pythia-70m
  
  # Check system requirements
  nanoaccel --check-requirements --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
        """
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model", 
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model identifier (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
    )
    model_group.add_argument(
        "--draft-model",
        default="EleutherAI/pythia-70m",
        help="Draft model for speculative decoding (default: EleutherAI/pythia-70m)"
    )
    
    # Quantization options
    quant_group = parser.add_argument_group("Quantization Options")
    quant_group.add_argument(
        "--quant",
        choices=["int2", "int4", "int8"],
        help="Enable quantization (int2/int4/int8)"
    )
    quant_group.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision inference"
    )
    
    # Generation options
    gen_group = parser.add_argument_group("Generation Options")
    gen_group.add_argument(
        "--prompt",
        default="Hello, world!",
        help="Input prompt text (default: 'Hello, world!')"
    )
    gen_group.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum number of tokens to generate (default: 50)"
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0)"
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter (default: 0.9)"
    )
    gen_group.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter (default: 50)"
    )
    gen_group.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (use greedy decoding)"
    )
    
    # Speculative decoding options
    spec_group = parser.add_argument_group("Speculative Decoding Options")
    spec_group.add_argument(
        "--speculative",
        action="store_true",
        help="Enable speculative decoding"
    )
    spec_group.add_argument(
        "--gamma",
        type=int,
        default=4,
        help="Number of speculative tokens (default: 4)"
    )
    spec_group.add_argument(
        "--early-exit-threshold",
        type=float,
        default=0.9,
        help="Early exit threshold for draft generation (default: 0.9)"
    )
    
    # System options
    system_group = parser.add_argument_group("System Options")
    system_group.add_argument(
        "--cpu-optimization",
        action="store_true",
        default=True,
        help="Enable CPU-specific optimizations (default: True)"
    )
    system_group.add_argument(
        "--no-cpu-optimization",
        action="store_false",
        dest="cpu_optimization",
        help="Disable CPU-specific optimizations"
    )
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        help="Output file for generated text"
    )
    output_group.add_argument(
        "--stats",
        action="store_true",
        help="Show detailed performance statistics"
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    output_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output except results"
    )
    
    # Utility options
    util_group = parser.add_argument_group("Utility Options")
    util_group.add_argument(
        "--check-requirements",
        action="store_true",
        help="Check system requirements for the specified model"
    )
    util_group.add_argument(
        "--cpu-info",
        action="store_true",
        help="Display CPU information"
    )
    util_group.add_argument(
        "--version",
        action="version",
        version="NanoAccel 0.1.0"
    )
    
    return parser


def check_requirements(args) -> None:
    """Check system requirements and exit."""
    meets_requirements, message = check_system_requirements(
        args.model, 
        args.quant, 
        sequence_length=2048
    )
    
    if args.json:
        result = {
            "meets_requirements": meets_requirements,
            "message": message,
            "model": args.model,
            "quantization": args.quant
        }
        print(json.dumps(result, indent=2))
    else:
        status = "✓" if meets_requirements else "✗"
        print(f"{status} System Requirements Check")
        print(f"Model: {args.model}")
        print(f"Quantization: {args.quant or 'None'}")
        print(f"Result: {message}")
    
    sys.exit(0 if meets_requirements else 1)


def display_cpu_info() -> None:
    """Display CPU information and exit."""
    cpu_info = detect_cpu_features()
    
    print("CPU Information:")
    print(f"  Cores: {cpu_info['cores']}")
    print(f"  AVX2: {'Yes' if cpu_info['avx2'] else 'No'}")
    print(f"  AVX512: {'Yes' if cpu_info['avx512'] else 'No'}")
    print(f"  SSE4: {'Yes' if cpu_info['sse4'] else 'No'}")
    
    if cpu_info['frequency']:
        freq = cpu_info['frequency']
        print(f"  Frequency: {freq.get('current', 'N/A')} MHz")
        print(f"  Min/Max: {freq.get('min', 'N/A')}-{freq.get('max', 'N/A')} MHz")
    
    if cpu_info['cache_size']:
        cache = cpu_info['cache_size']
        for level, size in cache.items():
            if size:
                print(f"  {level} Cache: {size}")
    
    sys.exit(0)


def setup_logging(verbose: bool, quiet: bool) -> None:
    """Setup logging configuration."""
    if quiet:
        logging.basicConfig(level=logging.ERROR)
    elif verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    
    # Handle utility commands
    if args.check_requirements:
        check_requirements(args)
    
    if args.cpu_info:
        display_cpu_info()
    
    # Initialize NanoAccel
    try:
        # Create quantization config if specified
        quant_config = None
        if args.quant:
            quant_config = QuantizationConfig(
                enabled=True,
                quant_type=args.quant,
                compute_dtype=torch.bfloat16 if args.mixed_precision else torch.float32
            )
        
        # Initialize NanoAccel
        nanoaccel = NanoAccel(
            model_name=args.model,
            quant_config=quant_config,
            mixed_precision=args.mixed_precision,
            cpu_optimization=args.cpu_optimization,
            verbose=args.verbose and not args.quiet
        )
        
        # Load model
        if not args.quiet:
            print(f"Loading model: {args.model}")
        
        nanoaccel.load_model()
        
        # Load draft model if using speculative decoding
        if args.speculative:
            if not args.quiet:
                print(f"Loading draft model: {args.draft_model}")
            nanoaccel.load_draft_model(args.draft_model)
        
        # Generate text
        if not args.quiet:
            print("Generating text...")
        
        result = nanoaccel.generate(
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=not args.no_sample,
            use_speculative=args.speculative,
            gamma=args.gamma,
            early_exit_threshold=args.early_exit_threshold
        )
        
        # Output results
        if args.json:
            output_data = {
                "text": result["text"],
                "tokens_generated": result["tokens_generated"],
                "generation_time": result["generation_time"],
                "tokens_per_second": result["tokens_per_second"],
                "method": result["method"]
            }
            
            if args.stats:
                output_data["stats"] = nanoaccel.get_stats()
                if args.speculative:
                    output_data["speculative_stats"] = nanoaccel.speculative_decoder.get_stats()
            
            print(json.dumps(output_data, indent=2))
        else:
            if not args.quiet:
                print(f"\nGenerated {result['tokens_generated']} tokens in "
                      f"{result['generation_time']:.2f}s "
                      f"({result['tokens_per_second']:.2f} tokens/sec)")
                print(f"Method: {result['method']}")
                print("\nGenerated text:")
                print("-" * 50)
            
            print(result["text"])
            
            if args.stats:
                stats = nanoaccel.get_stats()
                print(f"\nStatistics:")
                print(f"  Total tokens generated: {stats['total_tokens']}")
                print(f"  Total time: {stats['total_time']:.2f}s")
                print(f"  Average tokens/sec: {stats.get('average_tokens_per_second', 0):.2f}")
        
        # Save to file if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result["text"])
            if not args.quiet:
                print(f"\nOutput saved to: {args.output}")
    
    except KeyboardInterrupt:
        if not args.quiet:
            print("\nGeneration interrupted by user")
        sys.exit(1)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
