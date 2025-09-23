#!/usr/bin/env python3
"""
Comprehensive Research-Grade Validation Suite
Runs all validation tests for scientific certification.
"""

import sys
import time
import subprocess
import logging

# Configure logging for clean output
logging.basicConfig(level=logging.WARNING)

def run_test_suite():
    """Run the complete research-grade validation test suite."""
    
    print("🔬 COMPREHENSIVE RESEARCH-GRADE VALIDATION SUITE")
    print("=" * 70)
    print("Running complete test battery for scientific certification...")
    print()
    
    # Test suite configuration
    tests = [
        {
            "name": "Core Physics Validation",
            "script": "research_grade_validation.py",
            "description": "Basic physics accuracy and system integration",
            "critical": True
        },
        {
            "name": "Long-Term Stability",
            "script": "long_term_stability_test.py", 
            "description": "Extended propagation numerical stability",
            "critical": True
        },
        {
            "name": "Cross-Validation",
            "script": "cross_validation_test.py",
            "description": "Validation against known orbital data",
            "critical": True
        },
        {
            "name": "Monte Carlo Uncertainty",
            "script": "monte_carlo_uncertainty_test.py",
            "description": "Robustness under parameter uncertainties",
            "critical": False
        }
    ]
    
    results = []
    start_time = time.time()
    
    # Run each test
    for i, test in enumerate(tests, 1):
        print(f"🧪 Test {i}/{len(tests)}: {test['name']}")
        print(f"   {test['description']}")
        print(f"   Running {test['script']}...")
        
        test_start = time.time()
        
        try:
            # Run the test script
            result = subprocess.run(
                [sys.executable, test['script']],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test
            )
            
            test_duration = time.time() - test_start
            
            if result.returncode == 0:
                status = "✅ PASSED"
                passed = True
            else:
                status = "❌ FAILED"
                passed = False
                print(f"   Error output: {result.stderr[:200]}...")
            
            print(f"   {status} ({test_duration:.1f}s)")
            
            results.append({
                'name': test['name'],
                'passed': passed,
                'critical': test['critical'],
                'duration': test_duration,
                'output': result.stdout if passed else result.stderr
            })
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ TIMEOUT (>5 minutes)")
            results.append({
                'name': test['name'],
                'passed': False,
                'critical': test['critical'],
                'duration': 300,
                'output': "Test timed out"
            })
            
        except FileNotFoundError:
            print(f"   ❌ SCRIPT NOT FOUND")
            results.append({
                'name': test['name'],
                'passed': False,
                'critical': test['critical'],
                'duration': 0,
                'output': f"Script {test['script']} not found"
            })
        
        print()
    
    # Generate comprehensive report
    total_duration = time.time() - start_time
    
    print("📊 COMPREHENSIVE VALIDATION REPORT")
    print("=" * 70)
    
    # Test results summary
    passed_tests = sum(1 for r in results if r['passed'])
    total_tests = len(results)
    critical_tests = [r for r in results if r['critical']]
    critical_passed = sum(1 for r in critical_tests if r['passed'])
    
    print(f"Overall Results: {passed_tests}/{total_tests} tests passed")
    print(f"Critical Tests: {critical_passed}/{len(critical_tests)} passed")
    print(f"Total Duration: {total_duration:.1f} seconds")
    print()
    
    # Detailed results
    print("Detailed Results:")
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        critical = "CRITICAL" if result['critical'] else "optional"
        print(f"  {result['name']:25s} {status} ({result['duration']:4.1f}s) [{critical}]")
    
    print()
    
    # Research-grade certification assessment
    all_critical_passed = all(r['passed'] for r in critical_tests)
    high_pass_rate = (passed_tests / total_tests) >= 0.75
    
    if all_critical_passed and high_pass_rate:
        certification = "🏆 RESEARCH-GRADE CERTIFIED"
        print(certification)
        print("✅ Simulator meets scientific research standards")
        print("✅ Suitable for publication-quality studies")
        print("✅ Validated for orbital mechanics research")
        
        # Additional quality metrics
        if passed_tests == total_tests:
            print("🌟 GOLD STANDARD: All tests passed!")
        
    elif all_critical_passed:
        certification = "⚠️  RESEARCH-GRADE (with limitations)"
        print(certification)
        print("✅ Core physics validated for research use")
        print("⚠️  Some optional validations failed")
        print("📋 Review failed tests before publication")
        
    else:
        certification = "❌ NOT RESEARCH-GRADE"
        print(certification)
        print("❌ Critical validation failures detected")
        print("🔧 Address failed tests before research use")
        print("📋 Simulator requires improvements")
    
    print()
    print("=" * 70)
    
    # Save detailed report
    with open("validation_report.txt", "w") as f:
        f.write("COMPREHENSIVE RESEARCH-GRADE VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_duration:.1f} seconds\n")
        f.write(f"Overall Results: {passed_tests}/{total_tests} tests passed\n")
        f.write(f"Certification: {certification}\n\n")
        
        for result in results:
            f.write(f"Test: {result['name']}\n")
            f.write(f"Status: {'PASSED' if result['passed'] else 'FAILED'}\n")
            f.write(f"Duration: {result['duration']:.1f}s\n")
            f.write(f"Critical: {result['critical']}\n")
            f.write("Output:\n")
            f.write(result['output'][:1000] + "\n")
            f.write("-" * 40 + "\n\n")
    
    print("📄 Detailed report saved to: validation_report.txt")
    
    return all_critical_passed

if __name__ == "__main__":
    success = run_test_suite()
    sys.exit(0 if success else 1)