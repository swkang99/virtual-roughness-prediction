import subprocess
import sys
from pathlib import Path

# CI-like wrapper: ensure valid cache exists then run val.py

def main():
    config = Path('config.yaml')
    # first check cache for valid (this will create missing cache by default)
    cmd_check = [sys.executable, 'check_feature_cache.py', '--config', str(config), '--splits', 'valid']
    print('Checking/creating valid feature cache...')
    subprocess.check_call(cmd_check)

    # now run validation
    cmd_val = [sys.executable, 'val.py', '--config', str(config)]
    print('Starting validation...')
    subprocess.check_call(cmd_val)

if __name__ == '__main__':
    main()
