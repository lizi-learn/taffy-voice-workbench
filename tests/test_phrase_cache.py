import importlib.util
from pathlib import Path
import tempfile
import unittest
import wave

spec=importlib.util.spec_from_file_location('phrase_cache',Path(__file__).resolve().parents[1]/'edge/phrase_cache.py')
cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)

class PhraseCacheTests(unittest.TestCase):
    def test_identity_and_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); wav=root/'source.wav'
            with wave.open(str(wav),'wb') as f:
                f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(b'\0\0'*160)
            out=cache.put(root/'cache','你好','profile-v1',wav)
            self.assertEqual(cache.digest(wav),cache.digest(cache.get(root/'cache','你好','profile-v1')))
            with self.assertRaises(FileNotFoundError):cache.get(root/'cache','你好','profile-v2')
            out.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):cache.get(root/'cache','你好','profile-v1')

    def test_no_silent_text_normalization(self):
        self.assertNotEqual(cache.key('你好','v1'),cache.key('你好！','v1'))

if __name__=='__main__':unittest.main()
