import importlib.util
from pathlib import Path
import unittest
module=Path(__file__).resolve().parents[1]/'edge/runtime_bert.py'
spec=importlib.util.spec_from_file_location('edge_runtime',module)
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)

class EdgeContracts(unittest.TestCase):
    def test_window_coverage(self):
        for length in [1,32,64,65,173,593,3000]:
            windows=list(runtime.frame_windows(length,64,16))
            self.assertEqual(list(range(length)),[i for start,end,_,_ in windows for i in range(start,end)])
            for start,end,left,right in windows:
                self.assertLessEqual(left,start);self.assertGreaterEqual(right,end)
                self.assertGreaterEqual(left,0);self.assertLessEqual(right,length)
    def test_reject_inadequate_context(self):
        for args in [(0,64,16),(3,0,16),(100,64,8)]:
            with self.assertRaises(ValueError):list(runtime.frame_windows(*args))
    def test_no_silent_latin_loss(self):
        self.assertEqual(runtime.check_supported_text('你好，今天也要开心。'),[])
        with self.assertRaises(ValueError):runtime.check_supported_text('打开3D和YQ')
        self.assertEqual(runtime.check_supported_text('打开3D和YQ',True),['D','YQ'])

if __name__=='__main__':unittest.main()
