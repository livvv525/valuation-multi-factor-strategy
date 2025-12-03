"""
Basic tests for the valuation strategy
"""

def test_structure():
    """Test that project has correct structure"""
    import os
    
    required_folders = ['src', 'notebooks', 'config', 'docs', 'tests', 'results/images']
    required_files = [
        'README.md',
        'requirements.txt',
        '.gitignore',
        'src/valuation_strategy.py',
        'config/config.yaml'
    ]
    
    print("Checking project structure...")
    
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"✓ Folder exists: {folder}")
        else:
            print(f"✗ Missing folder: {folder}")
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ File exists: {file}")
        else:
            print(f"✗ Missing file: {file}")
    
    print("\nTest completed!")

if __name__ == '__main__':
    test_structure()