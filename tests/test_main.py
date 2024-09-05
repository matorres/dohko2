import subprocess

def test_greet():
    result = subprocess.run(['mycli', 'greet', '--name', 'Tester'], capture_output=True, text=True)
    assert result.stdout.strip() == "Hello, Tester!"