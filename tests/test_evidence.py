import unittest
from retrieval.evidence import fragments, audit_claims, normalized_numbers, group_similar


def document(text="本装置包括2个阀门，工作压力为3 MPa。采用过滤膜改善水质。", doc_id="CN-TEST"):
    h={"doc_id":doc_id,"abstract":text};h["evidence"]=fragments(h);return h


class EvidenceTests(unittest.TestCase):
    def test_exact_original_slices_and_stable_ids(self):
        h=document("  第一项技术。\n第二项；第三项。")
        for e in h["evidence"]:
            self.assertEqual(e["text"],h["abstract"][e["start"]:e["end"]])
        self.assertEqual(h["evidence"],fragments(h))

    def test_small_counts_are_not_skipped(self):
        h=document();claim={"doc_id":h["doc_id"],"text":"装置有3个阀门。","evidence_ids":[h["evidence"][0]["id"]]}
        self.assertEqual(audit_claims([claim],[h])["claims"][0]["status"],"numeric_review")

    def test_units_must_match(self):
        h=document();claim={"doc_id":h["doc_id"],"text":"工作压力3 kPa。","evidence_ids":[h["evidence"][0]["id"]]}
        self.assertEqual(audit_claims([claim],[h])["claims"][0]["status"],"numeric_review")

    def test_invalid_and_cross_document_references_blocked(self):
        a,b=document(),document(doc_id="CN-OTHER")
        for ids in [["made-up"],[b["evidence"][0]["id"]],[]]:
            c={"doc_id":a["doc_id"],"text":"采用阀门。","evidence_ids":ids}
            self.assertEqual(audit_claims([c],[a,b])["claims"][0]["status"],"invalid_reference")

    def test_valid_reference_does_not_claim_semantic_verification(self):
        h=document();c={"doc_id":h["doc_id"],"text":"装置包括2个阀门，工作压力3 MPa。","evidence_ids":[h["evidence"][0]["id"]]}
        r=audit_claims([c],[h]);self.assertEqual(r["claims"][0]["status"],"evidence_located");self.assertFalse(r["summary"]["semantic_verification"])

    def test_grouping_preserves_applications(self):
        hits=[document(doc_id="A"),document(doc_id="B")];self.assertEqual(group_similar(hits),1);self.assertEqual(len(hits),2)

    def test_zero_and_percentage(self):
        self.assertIn(("0","%"),normalized_numbers("浓度0%，误差0.5%。"));self.assertIn(("0.5","%"),normalized_numbers("误差0.5%。"))

    def test_missing_document_identity_is_not_a_valid_claim(self):
        h=document();c={"text":"采用阀门","evidence_ids":[h["evidence"][0]["id"]]}
        self.assertEqual(audit_claims([c],[h])["claims"][0]["status"],"invalid_reference")

    def test_range_dashes_are_not_negative_values(self):
        self.assertEqual(normalized_numbers("海藻糖1%–20%"),normalized_numbers("海藻糖1%-20%"))

    def test_astral_characters_keep_codepoint_offsets(self):
        h=document("𠀀型材料。第二个片段。")
        for e in h["evidence"]:self.assertEqual(h["abstract"][e["start"]:e["end"]],e["text"])


if __name__=="__main__":unittest.main()
