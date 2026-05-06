from __future__ import annotations
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ClinicalState
from ..agents.intake_agent import intake_agent
from ..agents.diagnosis_agent import diagnosis_agent
from ..agents.treatment_agent import treatment_agent
from ..agents.coding_agent import coding_agent
from ..agents.audit_agent import audit_agent



def _route_after_diagnosis(state:ClinicalState)->str:
    if state.needs_more_info:
        return "intake"
    return "treatment"

def build_clinical_pipeline(checkpointer=None):
    workflow=StateGraph(ClinicalState)

    # --- Register nodes ---
    workflow.add_node("intake",intake_agent)
    workflow.add_node("diagnosis",diagnosis_agent)
    workflow.add_node("treatment", treatment_agent)
    workflow.add_node("coding", coding_agent)
    workflow.add_node("audit", audit_agent)


    # --- Define edges ---
    workflow.set_entry_point("intake")
    workflow.add_edge(start_key="intake",end_key="diagnosis")

    workflow.add_conditional_edges(
        "diagnosis",
        _route_after_diagnosis,
        {
            "intake":"intake",
            "treatment":"treatment",
        },
    )

    workflow.add_edge("treatment", "coding")
    workflow.add_edge("coding", "audit")
    workflow.add_edge("audit", END)

    if checkpointer is None:
        checkpointer=MemorySaver()

    return workflow.compile(checkpointer=checkpointer)

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline=build_clinical_pipeline()
    return _pipeline