import tempfile
from pathlib import Path
import sys
import unittest
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workbench import Request, MOODS, VECTORS, INSTRUCTIONS
from timbre import GsvParameters, BertParameters, neutral_only

class ContractTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ref = Path(self.tmp.name) / "reference.wav"
        self.ref.touch()
        self.out = Path(self.tmp.name) / "output.wav"
    def tearDown(self):
        self.tmp.cleanup()
    def req(self, **kw):
        return Request(text=kw.pop("text", "这是一段人工智能合成的语音。"), reference=self.ref,
            reference_text="参考", output=self.out, **kw)
    def test_all_moods(self):
        for mood in MOODS:
            self.req(mood=mood).validate()
            self.assertEqual(8, len(VECTORS[mood]))
            self.assertLessEqual(sum(VECTORS[mood]), .8)
            self.assertIn(mood, INSTRUCTIONS)
    def test_reject_bad_inputs(self):
        for kw in [dict(text=""), dict(text="a"*301), dict(strength=1.01), dict(speed=0), dict(mood="unknown")]:
            with self.assertRaises(ValueError):
                self.req(**kw).validate()
    def test_no_overwrite(self):
        self.out.touch()
        with self.assertRaises(FileExistsError):
            self.req().validate()
    def test_no_instruction_in_spoken_text(self):
        r = self.req(mood="happy")
        self.assertNotIn("请非常开心", r.text)
    def test_engine_specific_prompts(self):
        rows=json.loads((Path(__file__).resolve().parents[1]/'prompts.json').read_text())
        self.assertEqual(21,len(rows))
        self.assertEqual(18,sum('cosy' in r.get('engines',['cosy','index']) for r in rows))
    def test_gsv_parameters(self):
        GsvParameters(auxiliary_references=(self.ref,)).validate()
        for kw in [dict(top_k=0),dict(top_k=1.5),dict(top_k=True),dict(top_p=0),dict(temperature=0),dict(temperature=float('nan')),dict(auxiliary_references=(self.out,))]:
            with self.assertRaises(ValueError):
                GsvParameters(**kw).validate()
    def test_bert_parameters(self):
        BertParameters(noise_scale=.35).validate()
        for kw in [dict(sdp_ratio=-1),dict(noise_scale=2),dict(noise_scale_w=float('nan'))]:
            with self.assertRaises(ValueError):
                BertParameters(**kw).validate()
    def test_no_fake_timbre_emotion(self):
        neutral_only(self.req())
        for kw in [dict(mood='happy'),dict(strength=.5)]:
            with self.assertRaises(ValueError):
                neutral_only(self.req(**kw))
    def test_bert_without_reference(self):
        req=Request('人工智能合成。',None,'',self.out)
        req.validate(require_reference=False)
        with self.assertRaises(ValueError):
            req.validate()

if __name__ == "__main__":
    unittest.main()
