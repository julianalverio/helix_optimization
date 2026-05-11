"""Entry point — wire the Designer, ParetoFrontier, metrics, and Claude
client into the optimization loop.

Run: `python -m twistr.agent.driver --config runtime/configs/agent.yaml`
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

from .claude_client import ClaudeClient
from .config import AgentConfig, load_agent_config
from .designer import Designer, THREE_LETTER
from .metrics import (
    buried_surface_area,
    electrostatic_complementarity,
    shape_complementarity,
)
from .pareto import Design, ParetoFrontier
from .prompts import SYSTEM_PROMPT, initial_user_prompt
from .tools import TOOLS

logger = logging.getLogger("twistr.agent")


def _score(designer: Designer, cfg: AgentConfig) -> tuple[float, float, float]:
    """Run one forward pass and compute (SC, EC, BSA)."""
    pred = designer.predict()
    sc = shape_complementarity(
        pred.atoms_atom14_ang, pred.atom_mask, pred.is_helix,
        pred.is_interface_residue, sigma=cfg.sc_distance_sigma,
    )
    ec = electrostatic_complementarity(
        pred.atoms_atom14_ang, pred.atom_mask, pred.residue_type,
        pred.is_helix, pred.is_interface_residue,
        distance_cutoff=cfg.ec_distance_cutoff,
    )
    bsa = buried_surface_area(
        pred.atoms_atom14_ang, pred.atom_mask, pred.residue_type,
        pred.is_helix, probe_radius=cfg.bsa_probe_radius,
    )
    return sc, ec, bsa


class AgentState:
    """Mutable state threaded through tool executions."""

    def __init__(self, designer: Designer, pareto: ParetoFrontier, cfg: AgentConfig):
        self.designer = designer
        self.pareto = pareto
        self.cfg = cfg
        self.iteration = 0
        self.next_id = 0
        self.mutation_history: list[tuple[str, int, str, str]] = []
        """(chain_id, position, from_aa, to_aa) sequence from WT to current state."""
        self.designs_by_id: dict[int, Design] = {}
        self.finished = False
        self.finish_summary = ""

    def register_design(self, design: Design) -> None:
        self.designs_by_id[design.id] = design

    def fresh_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1


def _execute_tool(name: str, args: dict, state: AgentState) -> dict:
    """Dispatch a Claude tool call to the appropriate handler and return
    a JSON-serializable result dict."""
    if name == "propose_mutation":
        return _tool_propose_mutation(args, state)
    if name == "query_residue":
        return _tool_query_residue(args, state)
    if name == "report_pareto":
        return _tool_report_pareto(args, state)
    if name == "revert_to_design":
        return _tool_revert(args, state)
    if name == "finish":
        return _tool_finish(args, state)
    return {"error": f"unknown tool {name!r}"}


def _tool_propose_mutation(args: dict, state: AgentState) -> dict:
    chain = args["helix_chain"]
    position = int(args["position"])
    new_aa = args["new_residue"].upper()
    rationale = args.get("rationale", "")

    try:
        from_aa = state.designer.residue_at(chain, position)
        state.designer.apply_mutation(chain, position, new_aa)
    except ValueError as e:
        # Don't consume an iteration on argument-validation failures.
        return {
            "error": str(e),
            "remaining_iterations": state.cfg.max_iterations - state.iteration,
        }

    state.mutation_history.append((chain, position, from_aa, new_aa))
    state.iteration += 1

    sc, ec, bsa = _score(state.designer, state.cfg)

    design = Design(
        id=state.fresh_id(),
        parent_id=None,
        mutations=list(state.mutation_history),
        helix_sequences=state.designer.helix_sequences(),
        sc=sc, ec=ec, bsa=bsa,
        iteration=state.iteration,
        notes=rationale,
    )
    on_frontier = state.pareto.add(design)
    state.register_design(design)

    # Write a PDB for every accepted Pareto design so the final
    # deliverable is structures, not just sequences.
    if on_frontier:
        out_dir = Path(state.cfg.output_dir) / "designs"
        state.designer.write_pdb(out_dir / f"design_{design.id:04d}.pdb")

    if (state.iteration % state.cfg.save_intermediate_every) == 0:
        _persist(state)

    return {
        "design_id": design.id,
        "sc": sc, "ec": ec, "bsa": bsa,
        "admitted_to_pareto": on_frontier,
        "frontier_size": len(state.pareto),
        "current_sequence": state.designer.helix_sequences(),
        "iteration": state.iteration,
        "remaining_iterations": state.cfg.max_iterations - state.iteration,
    }


def _tool_query_residue(args: dict, state: AgentState) -> dict:
    try:
        aa = state.designer.residue_at(args["helix_chain"], int(args["position"]))
    except ValueError as e:
        return {"error": str(e)}
    return {"residue": aa}


def _tool_report_pareto(args: dict, state: AgentState) -> dict:
    return {
        "frontier_size": len(state.pareto),
        "designs": [dataclasses.asdict(d) for d in state.pareto.designs],
    }


def _tool_revert(args: dict, state: AgentState) -> dict:
    target_id = int(args["design_id"])
    target = state.designs_by_id.get(target_id)
    if target is None:
        return {"error": f"no design with id {target_id}"}

    # Rebuild the mutation history from WT residue types for every
    # affected (chain, position), then re-apply only the mutations on
    # the target's lineage.
    for chain, position, from_aa, _to_aa in reversed(state.mutation_history):
        state.designer.apply_mutation(chain, position, from_aa)
    state.mutation_history = []
    for chain, position, _from_aa, to_aa in target.mutations:
        state.designer.apply_mutation(chain, position, to_aa)
    state.mutation_history = list(target.mutations)
    return {
        "reverted_to": target.id,
        "current_sequence": state.designer.helix_sequences(),
        "metrics_at_revert_point": {"sc": target.sc, "ec": target.ec, "bsa": target.bsa},
    }


def _tool_finish(args: dict, state: AgentState) -> dict:
    state.finished = True
    state.finish_summary = args.get("summary", "")
    return {"acknowledged": True, "final_frontier_size": len(state.pareto)}


def _persist(state: AgentState) -> Path:
    out_dir = Path(state.cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pareto_path = out_dir / "pareto_current.jsonl"
    state.pareto.to_jsonl(pareto_path)
    return pareto_path


def run(cfg: AgentConfig) -> ParetoFrontier:
    """Run the agent loop. Returns the final Pareto frontier."""
    logger.info("loading target+helices from %s", cfg.target_pdb_path)
    designer = Designer(
        target_pdb=Path(cfg.target_pdb_path),
        helix_chain_ids=cfg.helix_chain_ids,
        target_chain_ids=cfg.target_chain_ids,
        checkpoint_path=Path(cfg.checkpoint_path),
        device=cfg.device,
        interface_contact_distance=cfg.interface_contact_distance,
    )

    pareto = ParetoFrontier()
    state = AgentState(designer, pareto, cfg)

    # Seed the frontier with WT.
    wt_sc, wt_ec, wt_bsa = _score(designer, cfg)
    wt = Design(
        id=state.fresh_id(),
        parent_id=None,
        mutations=[],
        helix_sequences=designer.helix_sequences(),
        sc=wt_sc, ec=wt_ec, bsa=wt_bsa,
        iteration=0, notes="wild-type baseline",
    )
    pareto.add(wt)
    state.register_design(wt)
    designer.write_pdb(Path(cfg.output_dir) / "designs" / f"design_{wt.id:04d}.pdb")
    logger.info("WT scored: SC=%+0.3f EC=%+0.3f BSA=%.1f", wt_sc, wt_ec, wt_bsa)

    client = ClaudeClient(model=cfg.claude_model, max_tokens=cfg.max_tokens_per_turn)
    messages = [{
        "role": "user",
        "content": initial_user_prompt(
            designer.helix_sequences(), wt_sc, wt_ec, wt_bsa, cfg.max_iterations,
        ),
    }]

    while state.iteration < cfg.max_iterations and not state.finished:
        response = client.send(SYSTEM_PROMPT, messages, TOOLS)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            logger.info("Claude returned end_turn; stopping.")
            break

        tool_results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            result = _execute_tool(block.name, dict(block.input), state)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        if not tool_results:
            logger.warning("no tool calls in turn; stopping.")
            break

        messages.append({"role": "user", "content": tool_results})

    # Final persist.
    final_path = Path(cfg.output_dir) / f"pareto_final_{int(time.time())}.jsonl"
    state.pareto.to_jsonl(final_path)
    logger.info(
        "agent run complete: %d designs on Pareto frontier, %d total mutations evaluated; written to %s",
        len(state.pareto), state.iteration, final_path,
    )
    if state.finish_summary:
        logger.info("agent summary: %s", state.finish_summary)
    return state.pareto


def main() -> None:
    parser = argparse.ArgumentParser(prog="twistr-agent")
    parser.add_argument(
        "--config", type=str,
        default="runtime/configs/agent.yaml",
        help="Path to the agent YAML config.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_agent_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
