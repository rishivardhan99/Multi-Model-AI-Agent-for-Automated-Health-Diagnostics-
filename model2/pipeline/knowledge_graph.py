# model2/pipeline/knowledge_graph.py
"""
Lightweight knowledge graph implementation for deterministic reasoning.
(unchanged core; small additional edges added)
"""
from typing import Dict, List, Any

class KnowledgeGraph:
    def __init__(self):
        # graph: node -> list of edges
        # edge: {"relation": "suggests", "target": "Anemia", "weight": 0.9, "evidence": "..."}
        self.graph: Dict[str, List[Dict[str, Any]]] = {}

    def add_edge(self, source: str, relation: str, target: str, weight: float = 0.5, evidence: str = ""):
        self.graph.setdefault(source, []).append({
            "relation": relation,
            "target": target,
            "weight": float(weight),
            "evidence": evidence
        })

    def query(self, node: str, relation: str = None) -> List[Dict[str, Any]]:
        edges = self.graph.get(node, [])
        if relation:
            return [e for e in edges if e["relation"] == relation]
        return edges

    def infer_causes(self, observations: List[str]) -> Dict[str, float]:
        """
        Given a list of observation nodes (e.g., ["Platelets_LOW","Hemoglobin_LOW","MCV_LOW"])
        return a dict of {candidate_cause: aggregated_score}
        """
        scores: Dict[str, float] = {}
        for obs in observations:
            edges = self.query(obs, relation="possible_cause") + self.query(obs, relation="suggests")
            for e in edges:
                tgt = e["target"]
                # use max weight for simple evidence aggregation baseline
                scores[tgt] = max(scores.get(tgt, 0.0), e["weight"])
        # normalize to 0-1 by dividing by max if >1
        mx = max(scores.values()) if scores else 0.0
        if mx > 1.0:
            for k in list(scores.keys()):
                scores[k] = scores[k] / mx
        return scores

def build_medical_kg() -> KnowledgeGraph:
    kg = KnowledgeGraph()
    # Hematology
    kg.add_edge("Hemoglobin_LOW", "suggests", "Anemia", 0.9, evidence="Low haemoglobin suggests anemia")
    kg.add_edge("MCV_LOW", "possible_cause", "Iron_Deficiency", 0.65, evidence="Microcytic anemia often due to iron deficiency")
    kg.add_edge("RDW_HIGH", "possible_cause", "Iron_Deficiency", 0.6, evidence="High RDW supports iron deficiency")
    kg.add_edge("MCV_HIGH", "possible_cause", "Vitamin_B12_Deficiency", 0.75, evidence="Macrocytic pattern suggests B12/folate deficiency")

    kg.add_edge("Platelets_LOW", "suggests", "Thrombocytopenia", 0.95, evidence="Low platelets = thrombocytopenia")
    kg.add_edge("Thrombocytopenia", "possible_cause", "ITP", 0.7, evidence="Immune thrombocytopenia is a common cause of isolated low platelets")
    kg.add_edge("Thrombocytopenia", "possible_cause", "Viral_Infection", 0.6, evidence="Viruses can transiently lower platelets")
    kg.add_edge("Thrombocytopenia", "possible_cause", "Hypersplenism", 0.5, evidence="Splenic sequestration can lower platelets")

    # Inflammation / infection
    kg.add_edge("CRP_HIGH", "suggests", "Inflammation", 0.9, evidence="Raised CRP indicates acute inflammation")
    kg.add_edge("CRP_HIGH", "possible_cause", "Bacterial_Infection", 0.6, evidence="Raised CRP can support bacterial infection")
    kg.add_edge("Neutrophils_HIGH", "possible_cause", "Bacterial_Infection", 0.8, evidence="Neutrophilia suggests bacterial infection")
    kg.add_edge("Lymphocytes_HIGH", "possible_cause", "Viral_Infection", 0.7, evidence="Lymphocytosis can suggest viral processes")

    # Lipids and cardiovascular
    kg.add_edge("LDL_HIGH", "increases", "Cardiovascular_Risk", 0.85, evidence="High LDL increases atherogenic risk")
    kg.add_edge("Triglycerides_HIGH", "increases", "Cardiovascular_Risk", 0.6, evidence="High triglycerides associated with risk")
    kg.add_edge("TG_to_HDL_ratio_HIGH", "increases", "Cardiovascular_Risk", 0.9, evidence="High TG/HDL ratio indicates atherogenic dyslipidemia")
    kg.add_edge("HDL_LOW", "increases", "Cardiovascular_Risk", 0.7, evidence="Low HDL reduces protective effect")

    # Renal / metabolic
    kg.add_edge("Creatinine_HIGH", "suggests", "Kidney_Disease", 0.85, evidence="High creatinine may indicate impaired renal function")
    kg.add_edge("Urea_BUN_HIGH", "suggests", "Kidney_Disease", 0.6, evidence="Raised BUN can suggest renal dysfunction or prerenal causes")

    # glycaemic
    kg.add_edge("HbA1c_HIGH", "suggests", "Diabetes", 0.9, evidence="A1c >= 6.5 indicates diabetes in many guidelines")
    kg.add_edge("Glucose_Fasting_HIGH", "suggests", "Hyperglycemia", 0.8, evidence="Elevated fasting glucose suggests dysglycemia")

    return kg
