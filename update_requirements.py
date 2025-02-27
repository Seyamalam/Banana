#!/usr/bin/env python3
"""
Script to update requirements.txt by removing torchviz.
"""

import os

def update_requirements():
    """Update requirements.txt by removing torchviz."""
    # Read current requirements
    with open('requirements.txt', 'r') as f:
        requirements = f.readlines()
    
    # Filter out torchviz
    updated_requirements = [req for req in requirements if 'torchviz' not in req.lower()]
    
    # Write updated requirements
    with open('requirements.txt', 'w') as f:
        f.writelines(updated_requirements)
    
    print("Updated requirements.txt - removed torchviz dependency.")

if __name__ == "__main__":
    update_requirements() 