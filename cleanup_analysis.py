#!/usr/bin/env python3
"""
Cleanup Analysis and Removal Script

This script analyzes the remaining files and provides options for cleanup.
"""

import os
import shutil
from pathlib import Path

def analyze_remaining_files():
    """Analyze what files remain and their status"""
    
    print("🧹 CLEANUP ANALYSIS FOR REMAINING FILES")
    print("=" * 50)
    
    # Files that are now broken (import satellite_env which was moved)
    broken_files = [
        "complete_demo.py",
        "interactive_single_plot.py", 
        "quick_demo.py",
        "quick_visual_demo.py",
        "realtime_single_plot.py",
        "stable_monitor.py",
        "test_system.py", 
        "visual_training.py"
    ]
    
    # Documentation files (keep these)
    docs = [
        "README.md",
        "README_SATELLITE_RL.md", 
        "SUCCESS_VISUAL_SYSTEM.md",
        "VISUAL_GUIDE.md"
    ]
    
    # Image files (demo screenshots)
    images = [
        "mission_dashboard.png",
        "orbit_comparison.png", 
        "training_progress.png"
    ]
    
    # System directories
    directories = [
        "__pycache__",  # Can delete
        "orbit",        # EA Orbit framework - separate system
        "Geo_to_Leo"   # Our organized orbital transfer system
    ]
    
    print("📊 FILE ANALYSIS:")
    print("-" * 20)
    
    print(f"\n🚫 BROKEN FILES (import satellite_env that was moved):")
    for file in broken_files:
        if os.path.exists(file):
            print(f"   ❌ {file} - BROKEN, safe to delete")
        else:
            print(f"   ⚪ {file} - not found")
    
    print(f"\n📄 DOCUMENTATION FILES (keep these):")
    for file in docs:
        if os.path.exists(file):
            print(f"   ✅ {file} - KEEP (documentation)")
        else:
            print(f"   ⚪ {file} - not found")
    
    print(f"\n🖼️ IMAGE FILES (demo screenshots):")
    for file in images:
        if os.path.exists(file):
            print(f"   📸 {file} - KEEP (demo images)")
        else:
            print(f"   ⚪ {file} - not found")
    
    print(f"\n🗂️ DIRECTORIES:")
    for dir_name in directories:
        if os.path.exists(dir_name):
            if dir_name == "__pycache__":
                print(f"   🗑️ {dir_name}/ - DELETE (Python cache)")
            elif dir_name == "orbit":
                print(f"   🏭 {dir_name}/ - KEEP (EA Orbit framework)")
            elif dir_name == "Geo_to_Leo":
                print(f"   🛰️ {dir_name}/ - KEEP (our orbital system)")
            else:
                print(f"   📁 {dir_name}/ - exists")
        else:
            print(f"   ⚪ {dir_name}/ - not found")
    
    return broken_files, docs, images, directories

def cleanup_options():
    """Provide cleanup options"""
    
    broken_files, docs, images, directories = analyze_remaining_files()
    
    print(f"\n🎯 CLEANUP RECOMMENDATIONS:")
    print("-" * 30)
    print("✅ SAFE TO DELETE:")
    print("   • All broken .py files (they import moved satellite_env)")
    print("   • __pycache__/ directory (Python cache)")
    print("")
    print("✅ SAFE TO KEEP:")
    print("   • All README*.md files (documentation)")
    print("   • *.png files (demo screenshots)")
    print("   • orbit/ directory (EA's separate framework)")
    print("   • Geo_to_Leo/ directory (our orbital transfer system)")
    
    print(f"\n🤖 CLEANUP OPTIONS:")
    print("1. Delete only broken Python files")
    print("2. Delete broken files + __pycache__")
    print("3. Keep everything (no deletion)")
    print("4. Show what would be deleted (dry run)")
    
    try:
        choice = input("\nEnter choice (1-4): ").strip()
    except KeyboardInterrupt:
        print("\n⏹️ Cleanup cancelled")
        return
    
    if choice == "1":
        delete_broken_files(broken_files)
    elif choice == "2":
        delete_broken_files(broken_files)
        delete_pycache()
    elif choice == "3":
        print("✅ No files deleted - everything kept")
    elif choice == "4":
        dry_run_cleanup(broken_files)
    else:
        print("⚠️ Invalid choice")

def delete_broken_files(broken_files):
    """Delete broken Python files"""
    
    print(f"\n🗑️ DELETING BROKEN FILES:")
    deleted = 0
    
    for file in broken_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"   ✅ Deleted {file}")
                deleted += 1
            except Exception as e:
                print(f"   ❌ Failed to delete {file}: {e}")
        else:
            print(f"   ⚪ {file} not found")
    
    print(f"\n✅ Deleted {deleted} broken files")

def delete_pycache():
    """Delete Python cache directory"""
    
    if os.path.exists("__pycache__"):
        try:
            shutil.rmtree("__pycache__")
            print("   ✅ Deleted __pycache__/ directory")
        except Exception as e:
            print(f"   ❌ Failed to delete __pycache__: {e}")
    else:
        print("   ⚪ __pycache__ not found")

def dry_run_cleanup(broken_files):
    """Show what would be deleted without actually deleting"""
    
    print(f"\n🔍 DRY RUN - WOULD DELETE:")
    
    for file in broken_files:
        if os.path.exists(file):
            print(f"   🗑️ Would delete: {file}")
    
    if os.path.exists("__pycache__"):
        print(f"   🗑️ Would delete: __pycache__/")
    
    print(f"\n📋 WOULD KEEP:")
    keep_files = [
        "README.md", "README_SATELLITE_RL.md", "SUCCESS_VISUAL_SYSTEM.md", "VISUAL_GUIDE.md",
        "mission_dashboard.png", "orbit_comparison.png", "training_progress.png"
    ]
    
    for file in keep_files:
        if os.path.exists(file):
            print(f"   ✅ Would keep: {file}")
    
    print(f"   ✅ Would keep: orbit/ (EA Orbit framework)")
    print(f"   ✅ Would keep: Geo_to_Leo/ (our orbital system)")

if __name__ == "__main__":
    cleanup_options()