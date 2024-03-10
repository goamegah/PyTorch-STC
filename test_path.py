from pathlib import Path

HERE = Path(__file__).parent
print(HERE)
REQUIRED = [i.strip() for i in open(HERE / 'requirements.txt') if not i.startswith('#')]
print(REQUIRED)