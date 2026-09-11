import importlib.util,json,hashlib,tarfile,io
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('headline',Path(__file__).resolve().parents[1]/'scripts/reproduce.py')
repro=importlib.util.module_from_spec(spec);spec.loader.exec_module(repro)
def test_evidence_rejects_tampering(tmp_path):
    p=tmp_path/'a';p.write_bytes(b'original')
    manifest={'files':{'a':{'bytes':8,'sha256':hashlib.sha256(b'original').hexdigest()}}}
    repro.verify(tmp_path,manifest)
    p.write_bytes(b'modified')
    with pytest.raises(ValueError,match='mismatch'):repro.verify(tmp_path,manifest)
@pytest.mark.parametrize('name',['../escape','/absolute','C:/escape','x\\escape'])
def test_unsafe_evidence_path(tmp_path,name):
    with pytest.raises(ValueError):repro.safe_path(tmp_path,name)
def test_archive_rejects_link(tmp_path):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        i=tarfile.TarInfo('a');i.type=tarfile.SYMTYPE;i.linkname='../outside';t.addfile(i)
    with pytest.raises(ValueError):repro.unpack(archive,tmp_path/'out',{'files':{'a':{'bytes':0,'sha256':''}}})
def test_archive_requires_complete_manifest(tmp_path):
    archive=tmp_path/'empty.tar.gz'
    with tarfile.open(archive,'w:gz'):pass
    with pytest.raises(ValueError,match='Incomplete'):repro.unpack(archive,tmp_path/'out',{'files':{'missing':{'bytes':0,'sha256':''}}})
