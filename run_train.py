import subprocess
import sys
from pathlib import Path

# CI-like wrapper: ensure train cache exists then run train.py

def main():
    config = Path('config.yaml')
    # first check cache for train (this will create missing cache by default)
    cmd_check = [sys.executable, 'check_feature_cache.py', '--config', str(config), '--splits', 'train']
    print('Checking/creating train feature cache...')
    subprocess.check_call(cmd_check)

    # now run training
    cmd_train = [sys.executable, 'train.py', '--config', str(config)]
    print('Starting training...')
    subprocess.check_call(cmd_train)

if __name__ == '__main__':
    main()
