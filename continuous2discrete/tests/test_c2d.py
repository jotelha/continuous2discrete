import pytest
import subprocess

def test_pnp_output_format_txt(tmpdir):
    subprocess.run(
        ['pnp', '-c', '0.1', '0.1',
            '-u', '0.05', '-l', '1.0e-7', '-bc', 'cell', 'out.txt'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir)
