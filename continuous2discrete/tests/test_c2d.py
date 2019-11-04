import pytest
import subprocess
import numpy as np
import os.path

@pytest.fixture
def ref_npz(shared_datadir):
    """Provides 0.1 mM NaCl solution at 0.05 V acros 100 nm cell reference data from binary npz file"""
    return np.load( os.path.join(shared_datadir,'NaCl.npz') )

@pytest.fixture
def ref_txt(shared_datadir):
    """Provides 0.1 mM NaCl solution at 0.05 V acros 100 nm cell reference data from plain text file"""
    return np.loadtxt(os.path.join(shared_datadir,'NaCl.txt'), unpack=True)

def test_pnp_output_format_npz(tmpdir, ref_npz):
    print("  RUN test_pnp_output_format_npz")

    subprocess.run(
        ['pnp', '-c', '0.1', '0.1', '-z', '1', '-1',
            '-u', '0.05', '-l', '1.0e-7', '-bc', 'cell', 'out.npz'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir)

    test_npz = np.load(os.path.join(tmpdir,'out.npz'))
    dx = test_npz['x'] - ref_npz['x']
    du = test_npz['u'] - ref_npz['u']
    dc = test_npz['c'] - ref_npz['c']

    assert np.linalg.norm(dx) == 0
    assert np.linalg.norm(du) == 0
    assert np.linalg.norm(dc) == 0

def test_pnp_output_format_txt(tmpdir, ref_txt):
    print("  RUN test_pnp_output_format_txt")

    subprocess.run(
        ['pnp', '-c', '0.1', '0.1', '-z', '1', '-1',
            '-u', '0.05', '-l', '1.0e-7', '-bc', 'cell', 'out.txt'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=tmpdir)

    test_txt = np.loadtxt(os.path.join(tmpdir,'out.txt'), unpack=True)
    dx = test_txt[0,:] - ref_txt[0,:]
    du = test_txt[1,:] - ref_txt[1,:]
    dc = test_txt[2:,:] - ref_txt[2:,:]

    assert np.linalg.norm(dx) == 0
    assert np.linalg.norm(du) == 0
    assert np.linalg.norm(dc) == 0
