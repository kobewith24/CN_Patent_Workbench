import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"eval"))
from metrics import metrics


class MetricsTests(unittest.TestCase):
    def test_one_of_four_is_hit_not_full_recall(self):
        result=metrics(["a","x","y"],["a","b","c","d"],5)
        self.assertEqual(result["Hit@5"],1);self.assertEqual(result["Recall@5"],0.25)

    def test_duplicate_results_do_not_inflate_recall(self):
        self.assertEqual(metrics(["a","a"],["a","b"],5)["Recall@5"],0.5)


if __name__=="__main__":unittest.main()
