
import sys
import os

# Add the project root to sys.path explicitly
# This ensures that 'src' module can be found regardless of how pytest is run
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

print(f"Added to sys.path: {sys.path[0]}")
