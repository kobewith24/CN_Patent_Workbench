import unittest
from types import SimpleNamespace
from fastapi import HTTPException
from pydantic import ValidationError
from retrieval.app import Filters, Search, STATE, search


class FilterContractTests(unittest.TestCase):
    def test_unknown_hard_condition_is_not_ignored(self):
        with self.assertRaises(ValidationError):Filters(applicant_name="某公司")

    def test_inconsistent_conditions_are_rejected(self):
        for values in [{"year_from":2024,"year_to":2018},
                       {"ipc_section":"H","ipc_subclass":"G06T"},
                       {"patent_type":"未知类型"}]:
            with self.assertRaises(ValidationError):Filters(**values)

    def test_missing_reranker_is_explicit_failure(self):
        old=STATE.get("retriever")
        try:
            STATE["retriever"]=SimpleNamespace(reranker=None)
            with self.assertRaises(HTTPException) as error:search(Search(query="图像去噪"))
            self.assertEqual(error.exception.status_code,503)
        finally:
            if old is None:STATE.pop("retriever",None)
            else:STATE["retriever"]=old


if __name__=="__main__":unittest.main()
