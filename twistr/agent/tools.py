"""Anthropic tool-use schema for the lead-optimization agent."""
from __future__ import annotations

TOOLS = [
    {
        "name": "propose_mutation",
        "description": (
            "Apply a point mutation to one of the designed helices, re-run "
            "the lead-optimization model on the mutated complex, and return "
            "the predicted SC, EC, and BSA along with a flag indicating "
            "whether the new design entered the Pareto frontier. The "
            "mutation persists — subsequent propose_mutation calls compound "
            "on top of it unless you revert."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "helix_chain": {
                    "type": "string",
                    "description": "Chain ID of the helix to mutate (one of the two helix chain IDs).",
                },
                "position": {
                    "type": "integer",
                    "description": "Author-numbered residue position on the chain (1-indexed, matches the PDB).",
                },
                "new_residue": {
                    "type": "string",
                    "description": "Three-letter code of the new residue (e.g. 'LEU', 'PHE', 'ASP').",
                    "pattern": "^[A-Z]{3}$",
                },
                "rationale": {
                    "type": "string",
                    "description": "One-sentence justification recorded with the design (e.g. 'introduce charge complementarity at K42 paired with target E156').",
                },
            },
            "required": ["helix_chain", "position", "new_residue", "rationale"],
        },
    },
    {
        "name": "query_residue",
        "description": (
            "Return the current residue (three-letter code) at a given "
            "(chain, position) without mutating it. Use to confirm the "
            "starting state before proposing a mutation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "helix_chain": {"type": "string"},
                "position": {"type": "integer"},
            },
            "required": ["helix_chain", "position"],
        },
    },
    {
        "name": "report_pareto",
        "description": (
            "Return every design currently on the Pareto frontier, with "
            "its mutation history, helix sequences, and (SC, EC, BSA) "
            "scores. Use to plan reverts or to confirm the frontier has "
            "stabilized."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "revert_to_design",
        "description": (
            "Set the current state to a previously-scored design "
            "identified by its integer ID. Subsequent propose_mutation "
            "calls will branch from that design rather than from the "
            "latest attempt. Use to escape local optima."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "design_id": {
                    "type": "integer",
                    "description": "ID of the design to revert to (visible in report_pareto output).",
                },
            },
            "required": ["design_id"],
        },
    },
    {
        "name": "finish",
        "description": (
            "Stop the optimization loop. Use this when the Pareto frontier "
            "has stabilized and further mutations are not expanding it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "One-paragraph summary of the optimization trajectory and the final Pareto frontier.",
                },
            },
            "required": ["summary"],
        },
    },
]
