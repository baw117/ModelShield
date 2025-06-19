
"""
Simple startup script for ModelShield Server
Run this to start the server with default configuration
"""

import sys
import logging
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

def main():
    """Main startup function"""
    
    print("🛡️  ModelShield SERVER")
    print("=" * 50)
    
    try:

        from guardrails import ProductionGuardrailsServer, ProductionConfig
        

        config = ProductionConfig(
            enable_semantic_detection=True,     # Enable ML-based detection
            enable_enhanced_pii=True,           # Enable Presidio PII detection
            enable_output_scanning=True,        # Enable output validation
            enable_context_analysis=True,       # Enable context awareness
            enable_authentication=False,        # Disabled for demo/development
            enable_rate_limiting=True,          # Enable rate limiting
            log_level="INFO",                   # Set log level
            models_cache_dir="./models_cache",  # Cache ML models here
            logs_dir=Path("./logs"),            # Log files location
            data_dir=Path("./data")             # Data files location
        )
        
        print("📋 Configuration:")
        print(f"  • Semantic Detection: {config.enable_semantic_detection}")
        print(f"  • Enhanced PII: {config.enable_enhanced_pii}")
        print(f"  • Output Scanning: {config.enable_output_scanning}")
        print(f"  • Context Analysis: {config.enable_context_analysis}")
        print(f"  • Authentication: {config.enable_authentication}")
        print(f"  • Rate Limiting: {config.enable_rate_limiting}")
        print(f"  • Log Level: {config.log_level}")
        print()
        

        server = ProductionGuardrailsServer(config)
        
        print("🚀 Starting server on http://localhost:8000")
        print()
        print("📖 Available endpoints:")
        print("  • GET  /           - Server info")
        print("  • GET  /health     - Health check")
        print("  • POST /validate-input - Input validation")
        print("  • POST /validate-output - Output validation") 
        print("  • POST /process-complete - End-to-end processing")
        print("  • GET  /stats      - System statistics")
        print("  • GET  /system-info - Detailed system info")
        print("  • GET  /docs       - API documentation")
        print()
        print("🧪 Quick test:")
        print("  curl -X POST 'http://localhost:8000/validate-input' \\")
        print("       -H 'Content-Type: application/json' \\")
        print("       -d '{\"prompt\": \"Hello, how are you?\", \"user_id\": \"test\"}'")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        

        server.run(host="0.0.0.0", port=8000)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Make sure all required files are in the same directory:")
        print("  • guardrails.py")
        print("  • guardrails_server.py") 
        print("  • pii_detector.py")
        print("  • semantic_detector.py")
        print("  • output_scanner.py")
        print("  • comprehensive_test_guardrails.py")
        print("\n📦 Install required packages:")
        print("  pip install fastapi uvicorn presidio-analyzer presidio-anonymizer")
        print("  pip install transformers sentence-transformers torch")
        print("  pip install scikit-learn numpy pandas")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Check that all Python files are in the same directory")
        print("  2. Install missing dependencies")
        print("  3. Check file permissions")
        print("  4. Ensure port 8000 is available")
        sys.exit(1)

if __name__ == "__main__":
    main()