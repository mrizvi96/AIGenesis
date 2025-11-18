#!/usr/bin/env python3
"""
AI-Powered Insurance Claims Processing Assistant - Interactive Demo
Cloud-Optimized | Production Ready | Try It Now!

This script provides an interactive way to test the AI claims processing system.
Perfect for demonstrations, testing, and onboarding new users.

Usage:
    python demo.py                    # Interactive demo mode
    python demo.py --auto             # Automatic demo with sample data
    python demo.py --test             # Run integration tests
    python demo.py --install          # Quick installation check
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, List
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

# Color codes for beautiful output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_banner():
    """Print beautiful demo banner"""
    print(f"""
{Colors.HEADER}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   🏥🤖 AI-Powered Insurance Claims Processing Assistant                        ║
║                                                                              ║
║   ☁️  Cloud-Optimized | 🚀 Production Ready | 💡 Enterprise Scale          ║
║                                                                              ║
║   🌟 Try it now: https://github.com/mrizvi96/AIGenesis                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
{Colors.ENDC}
""")

def print_section(title: str, color: str = Colors.OKBLUE):
    """Print a section header"""
    print(f"\n{color}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{color}{Colors.BOLD}  {title}{Colors.ENDC}")
    print(f"{color}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_success(message: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠️  {message}{Colors.ENDC}")

def print_error(message: str):
    """Print error message"""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def print_info(message: str):
    """Print info message"""
    print(f"{Colors.OKCYAN}ℹ️  {message}{Colors.ENDC}")

def check_installation() -> bool:
    """Check if system is properly installed"""
    print_section("🔍 Installation Check")

    try:
        # Check Python version
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            print_error(f"Python {python_version.major}.{python_version.minor} detected. Python 3.8+ required.")
            return False
        print_success(f"Python {python_version.major}.{python_version.minor}.{python_version.micro} ✓")

        # Check required files
        required_files = [
            "backend/aiml_multi_task_classifier.py",
            "backend/qdrant_manager.py",
            "backend/memory_manager.py",
            "requirements.txt",
            ".env.example"
        ]

        for file in required_files:
            if not os.path.exists(file):
                print_error(f"Missing file: {file}")
                return False
            print_success(f"Found: {file}")

        # Check environment variables
        if not os.path.exists(".env"):
            print_warning("No .env file found. Creating from template...")
            try:
                import shutil
                shutil.copy(".env.example", ".env")
                print_success("Created .env from template")
                print_warning("Please edit .env with your Qdrant Cloud credentials")
            except Exception as e:
                print_error(f"Could not create .env file: {e}")
                return False
        else:
            print_success("Environment file (.env) found")

        # Try to import core modules
        try:
            os.environ.setdefault('PYTHONPATH', str(Path(__file__).parent))
            import dotenv
            dotenv.load_dotenv()
            print_success("Environment loaded successfully")
        except ImportError:
            print_warning("python-dotenv not installed. Install with: pip install python-dotenv")
        except Exception as e:
            print_warning(f"Environment loading issue: {e}")

        return True

    except Exception as e:
        print_error(f"Installation check failed: {e}")
        return False

def quick_health_check() -> Dict[str, Any]:
    """Quick system health check"""
    print_section("🏥 System Health Check")

    results = {
        'overall_status': 'unknown',
        'checks': {}
    }

    try:
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)

            results['checks']['memory'] = {
                'status': 'ok' if memory_gb >= 2 else 'warning',
                'total_gb': round(memory_gb, 2),
                'available_gb': round(available_gb, 2),
                'percent_used': memory.percent
            }

            if memory_gb >= 2:
                print_success(f"Memory: {round(memory_gb, 1)}GB total, {round(available_gb, 1)}GB available")
            else:
                print_warning(f"Memory: {round(memory_gb, 1)}GB total (recommended: 2GB+)")
        except ImportError:
            print_warning("psutil not available for memory checking")
            results['checks']['memory'] = {'status': 'unknown'}

        # Check disk space
        try:
            disk = psutil.disk_usage('.')
            disk_gb = disk.total / (1024**3)
            free_gb = disk.free / (1024**3)

            results['checks']['disk'] = {
                'status': 'ok' if free_gb >= 1 else 'warning',
                'total_gb': round(disk_gb, 2),
                'free_gb': round(free_gb, 2)
            }

            if free_gb >= 1:
                print_success(f"Disk: {round(free_gb, 1)}GB free")
            else:
                print_warning(f"Disk: {round(free_gb, 1)}GB free (recommended: 1GB+)")
        except:
            print_warning("Could not check disk space")
            results['checks']['disk'] = {'status': 'unknown'}

        # Check environment variables
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_key = os.getenv('QDRANT_API_KEY')

        if qdrant_url and qdrant_key:
            print_success("Qdrant Cloud credentials configured")
            results['checks']['qdrant_config'] = {'status': 'ok'}
        else:
            print_warning("Qdrant Cloud credentials not configured")
            results['checks']['qdrant_config'] = {'status': 'warning'}

        # Calculate overall status
        statuses = [check.get('status', 'unknown') for check in results['checks'].values()]
        if all(status == 'ok' for status in statuses):
            results['overall_status'] = 'excellent'
        elif all(status in ['ok', 'warning'] for status in statuses):
            results['overall_status'] = 'good'
        else:
            results['overall_status'] = 'needs_attention'

        return results

    except Exception as e:
        print_error(f"Health check failed: {e}")
        results['overall_status'] = 'error'
        return results

def run_sample_claim_processing() -> Dict[str, Any]:
    """Run sample claim processing demo"""
    print_section("🏥 Sample Claim Processing")

    # Sample claim data
    sample_claims = [
        {
            "id": "demo_001",
            "type": "Medical Emergency",
            "text": "Patient presents with severe chest pain radiating to left arm. Symptoms began approximately 45 minutes prior to arrival. Patient reports associated shortness of breath, diaphoresis, and nausea. ECG shows ST-segment elevation in leads II, III, aVF.",
            "amount": 15000,
            "priority": "High"
        },
        {
            "id": "demo_002",
            "type": "Vehicle Accident",
            "text": "Vehicle collision resulting in moderate front-end damage. Driver reports whiplash symptoms and minor cuts. Airbags deployed. Vehicle towed from scene. Police report filed.",
            "amount": 8500,
            "priority": "Medium"
        },
        {
            "id": "demo_003",
            "type": "Property Damage",
            "text": "Water damage to residential property due to burst pipe. Living room and kitchen affected. Drywall replacement needed. Estimated repair time: 2 weeks.",
            "amount": 12000,
            "priority": "Medium"
        }
    ]

    try:
        # Import the classifier
        print_info("Loading AI claim classifier...")
        from aiml_multi_task_classifier import get_aiml_multitask_classifier

        classifier = get_aiml_multitask_classifier()
        print_success("AI classifier loaded successfully")

        results = []

        for i, claim in enumerate(sample_claims, 1):
            print(f"\n{Colors.OKCYAN}Processing Claim {i}/{len(sample_claims)}: {claim['type']}{Colors.ENDC}")
            print(f"Amount: ${claim['amount']:,}")
            print(f"Priority: {claim['priority']}")
            print(f"Description: {claim['text'][:100]}...")

            try:
                start_time = time.time()

                # Process the claim
                result = classifier.classify_claim({
                    "claim_text": claim['text'],
                    "claim_type": claim['type'].lower(),
                    "amount": claim['amount']
                })

                processing_time = time.time() - start_time

                # Display results
                print(f"{Colors.OKGREEN}✅ Processing completed in {processing_time:.2f}s{Colors.ENDC}")

                if 'damage_assessment' in result:
                    print(f"  🎯 Damage Assessment: {result['damage_assessment']}")
                if 'fraud_probability' in result:
                    fraud_prob = result['fraud_probability']
                    risk_level = 'Low' if fraud_prob < 0.3 else 'Medium' if fraud_prob < 0.7 else 'High'
                    print(f"  🔍 Fraud Risk: {risk_level} ({fraud_prob:.2%})")
                if 'recommended_action' in result:
                    print(f"  📋 Recommended Action: {result['recommended_action']}")
                if 'urgency_level' in result:
                    print(f"  ⚡ Urgency: {result['urgency_level']}")

                results.append({
                    'claim_id': claim['id'],
                    'success': True,
                    'processing_time': processing_time,
                    'result': result
                })

            except Exception as e:
                print_error(f"Failed to process claim {claim['id']}: {e}")
                results.append({
                    'claim_id': claim['id'],
                    'success': False,
                    'error': str(e)
                })

        # Summary
        successful = sum(1 for r in results if r['success'])
        avg_time = sum(r['processing_time'] for r in results if r['success']) / max(successful, 1)

        print(f"\n{Colors.OKGREEN}{Colors.BOLD}📊 Processing Summary:{Colors.ENDC}")
        print(f"  ✅ Successful: {successful}/{len(results)} claims")
        print(f"  ⏱️  Average Time: {avg_time:.2f}s per claim")
        print(f"  🚀 Throughput: {60/avg_time:.1f} claims per hour")

        return {
            'total_claims': len(sample_claims),
            'successful': successful,
            'avg_processing_time': avg_time,
            'results': results
        }

    except ImportError as e:
        print_error(f"Could not import classifier: {e}")
        print_warning("Make sure dependencies are installed: pip install -r requirements.txt")
        return {'error': 'import_error', 'message': str(e)}
    except Exception as e:
        print_error(f"Claim processing failed: {e}")
        return {'error': 'processing_error', 'message': str(e)}

def run_integration_tests():
    """Run cloud integration tests"""
    print_section("🧪 Cloud Integration Tests")

    try:
        print_info("Running cloud integration tests...")
        from cloud_integration_test import CloudIntegrationTester

        tester = CloudIntegrationTester()
        results = tester.run_comprehensive_tests()

        print(f"\n{Colors.BOLD}{Colors.OKGREEN}🏆 Test Results Summary:{Colors.ENDC}")
        print(f"  🎯 Overall Success: {results.get('success_rate', 0):.1f}%")
        print(f"  ☁️  Cloud Ready: {'YES' if results.get('cloud_ready') else 'NO'}")
        print(f"  📊 Components Tested: {len(results.get('component_results', {}))}")

        if results.get('component_results'):
            print(f"\n{Colors.OKCYAN}Component Breakdown:{Colors.ENDC}")
            for component, result in results['component_results'].items():
                status = "✅" if result.get('success') else "❌"
                print(f"  {status} {component.replace('_', ' ').title()}")

        return results

    except ImportError as e:
        print_error(f"Could not import test module: {e}")
        return {'error': 'import_error'}
    except Exception as e:
        print_error(f"Integration tests failed: {e}")
        return {'error': 'test_error'}

def show_setup_guide():
    """Show setup guide"""
    print_section("🚀 Quick Setup Guide")

    guide = """
    📋 STEP-BY-STEP SETUP:

    1️⃣  INSTALL DEPENDENCIES:
        pip install -r requirements.txt

    2️⃣  SETUP QDRANT CLOUD (FREE):
        • Visit: https://cloud.qdrant.io/
        • Create free account
        • Create cluster (Free tier: 1GB RAM, 4GB storage)
        • Copy URL and API key

    3️⃣  CONFIGURE ENVIRONMENT:
        cp .env.example .env
        # Edit .env with your Qdrant credentials:
        QDRANT_URL=https://your-cluster.cloud.qdrant.io
        QDRANT_API_KEY=your-api-key

    4️⃣  RUN THE SYSTEM:
        python backend/main.py
        # Or run this demo: python demo.py

    5️⃣  TEST EVERYTHING:
        python backend/cloud_integration_test.py

    🎯 NEED HELP?
        • GitHub: https://github.com/mrizvi96/AIGenesis/issues
        • Email: mohammad.rizvi@csuglobal.edu
        • README: See README.md for detailed documentation
    """

    print(guide)

def interactive_menu():
    """Interactive demo menu"""
    while True:
        print_section("🎮 Interactive Demo Menu")

        options = [
            "1️⃣  Run System Health Check",
            "2️⃣  Process Sample Claims",
            "3️⃣  Run Cloud Integration Tests",
            "4️⃣  View Setup Guide",
            "0️⃣  Exit Demo"
        ]

        for option in options:
            print(f"  {option}")

        try:
            choice = input(f"\n{Colors.OKCYAN}Select an option (0-4): {Colors.ENDC}").strip()

            if choice == '0':
                print_success("Thank you for trying AI Claims Processing Assistant!")
                break
            elif choice == '1':
                quick_health_check()
            elif choice == '2':
                run_sample_claim_processing()
            elif choice == '3':
                run_integration_tests()
            elif choice == '4':
                show_setup_guide()
            else:
                print_warning("Invalid option. Please select 0-4.")

            input(f"\n{Colors.OKCYAN}Press Enter to continue...{Colors.ENDC}")

        except KeyboardInterrupt:
            print_success("\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print_error(f"Menu error: {e}")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="AI Claims Processing Assistant Demo")
    parser.add_argument("--auto", action="store_true", help="Run automatic demo")
    parser.add_argument("--test", action="store_true", help="Run integration tests only")
    parser.add_argument("--install", action="store_true", help="Check installation only")
    parser.add_argument("--health", action="store_true", help="Run health check only")

    args = parser.parse_args()

    print_banner()

    if args.install:
        if check_installation():
            print_success("\n🎉 Installation check passed! You're ready to go.")
        else:
            print_error("\n❌ Installation check failed. Please fix the issues above.")
        return

    if args.health:
        quick_health_check()
        return

    if args.test:
        run_integration_tests()
        return

    if args.auto:
        print_info("Running automatic demo...")

        # Installation check
        if not check_installation():
            return

        # Health check
        health = quick_health_check()

        # Sample claim processing
        claim_results = run_sample_claim_processing()

        # Integration tests
        test_results = run_integration_tests()

        # Final summary
        print_section("🎉 Demo Complete")
        print_success("Automatic demo completed successfully!")
        print_info("Try interactive mode with: python demo.py")

        return

    # Default: interactive mode
    if not check_installation():
        print_error("\nInstallation check failed. Please fix the issues before running the demo.")
        return

    interactive_menu()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_success("\n\nDemo interrupted. Goodbye! 👋")
    except Exception as e:
        print_error(f"Demo failed: {e}")
        print_info("For help, create an issue at: https://github.com/mrizvi96/AIGenesis/issues")

