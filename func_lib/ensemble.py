import uuid
from typing import Any, Dict, List, Optional, Tuple

from ..core import (
    AgentFunction,
    AgentNode,
    CodeFunction,
    FunctionArg,
    Node,
    Provider,
    RunContext,
    CancellationException,
)


class EnsembleException(Exception):
    def __init__(
        self,
        inner_ex: Dict[Provider, List[Exception]],
        instances: Dict[Provider, int],
    ):
        summary_parts: List[str] = []
        example_parts: List[str] = []
        for provider, total in instances.items():
            failures = len(inner_ex.get(provider, []))
            successes = total - failures
            summary_parts.append(f"{provider.value} success: {successes}/{total}")
            if failures:
                example = AgentNode.stringify_exception(inner_ex[provider][0])
                example_parts.append(
                    f"Example {provider.value} Exception: ({example})."
                )

        message = f"{', '.join(summary_parts)}; Fails `allow_fail` criteria."
        assert example_parts
        message = f"{message} {' '.join(example_parts)}"

        super().__init__(message)
        self.inner_exceptions = inner_ex

class Ensemble(CodeFunction):
    """
    Decorates an AgentFunction. For each invocation, fans out identical AgentFunction
    tasks and then uses the same agent with an injected reconciliation directive
    to reconcile the results. For analytical tasks, this is known as Ensemble-of-thought.
    For added efficacy, different model providers can also be used in parallel.

    - Signature mirrors the wrapped agent's args exactly.
    - Reconciles via the same AgentFunction, with modified prompt and tool discouragement.
    """
    def __init__(
        self,
        agent: AgentFunction,
        instances: Dict[Provider, int],
        allow_fail: Dict[Provider, int],
        name: Optional[str] = None,
        reconcile_by: Optional[Provider] = None,
    ) -> None:
        if not isinstance(agent, AgentFunction):
            raise TypeError("Ensemble requires an AgentFunction to decorate.")
        if not instances:
            raise ValueError("Ensemble requires at least one provider instance.")
        if not allow_fail:
            raise ValueError("'allow_fail' must be provided.")

        # Normalize + validate instances.
        norm: Dict[Provider, int] = {}
        for p, c in instances.items():
            if not isinstance(p, Provider):
                raise TypeError(f"'instances' keys must be Provider values, got {type(p).__name__}")
            n = int(c)
            if n > 0:
                norm[p] = n
        if not norm:
            raise ValueError("Ensemble requires at least one positive instance count.")

        # Normalize + validate allow_fail.
        allow: Dict[Provider, int] = {}
        for provider in norm:
            if provider not in allow_fail:
                raise ValueError(
                    "'allow_fail' must specify an allowed failure count for each provider in 'instances'."
                )
            fail_limit = int(allow_fail[provider])
            if fail_limit < 0:
                raise ValueError("'allow_fail' counts must be non-negative.")
            allow[provider] = fail_limit

        self._agent = agent
        self._instances = norm
        self._allow_fail = allow
        self._reconcile_provider: Provider = reconcile_by or agent.default_model

        fn_name = name or agent.name

        # Reconciliation agent.
        recon_name = f"{fn_name}__reconcile__{uuid.uuid4().hex[:8]}"  # Avoid BFS name collisions.
        self._reconcile_agent = AgentFunction(
            name=recon_name,
            desc=f"Reconcile ensemble candidates for '{agent.name}'.",
            args=[
                FunctionArg("original_user_prompt", str, "Original user prompt after substitution."),
                FunctionArg("ensemble_candidates", str, "Labeled candidate answers (markdown)."),
            ],
            system_prompt=agent.system_prompt,
            user_prompt_template=(
                "{original_user_prompt}\n\n"
                "--- Ensemble Candidates ---\n"
                "{ensemble_candidates}\n\n"
                "{reconciliation_prompt}\n"
            ),
            uses=agent.uses,
            default_model=self._reconcile_provider,
        )

        # Implementation (fan-out → gather → reconcile)
        def __impl(ctx: RunContext, arg_map: Dict[str, Any]) -> Any:
            self._agent.validate_coerce_args(arg_map)

            # Last-minute check for cancellation before we fan-out potentially
            # long-running tasks.
            if ctx.cancel_requested():
                raise CancellationException("Early request to cancel, before fan-out.")

            # Fan-out (parallel launch by Runtime).
            launches: List[Tuple[Provider, Node]] = []
            for prov, cnt in self._instances.items():
                for _ in range(cnt):
                    launches.append((prov, ctx.invoke(self._agent, arg_map, provider=prov)))
                    # No point checking cancellation again here because invoke() is fast.

            captured_ex: Dict[Provider, List[Exception]] = {
                prov: [] for prov in self._instances
            }

            # Gather candidates (robust to individual failures)
            candidates: List[str] = []
            excess_fail = False
            for idx, (prov, node) in enumerate(launches):
                try:
                    out = node.result()
                    text = "" if out is None else str(out).strip()
                except Exception as ex:
                    # Could be CancellationException if child was canceled, or the child decided
                    # to raise for any reason. Both cases are handled the same; we will check if
                    # we were also canceled after we finish collecting children results.
                    text = f"[ERROR] {type(ex).__name__}: {ex}"
                    captured_ex[prov].append(ex)
                    if len(captured_ex[prov]) > self._allow_fail[prov]:
                        excess_fail = True
                candidates.append(
                    f"### Candidate {idx}\n"
                    f"<candidate_answer idx={idx}>\n\n"
                    f"{text}\n\n"
                    f"</candidate_answer>")
            ensemble_candidates = "\n\n".join(candidates)

            if ctx.cancel_requested():
                raise CancellationException(
                    "Requested to cancel after children fan-out was initiated; before reconciliation.")
            
            if excess_fail:
                raise EnsembleException(captured_ex, self._instances)

            # Reconcile.
            recon_inputs = {
                "original_user_prompt": self._agent.user_prompt_template.format(**arg_map),
                "ensemble_candidates": ensemble_candidates,
                "reconciliation_prompt": reconciliation_prompt,
            }
            recon_node = ctx.invoke(self._reconcile_agent, recon_inputs, provider=self._reconcile_provider)
            try:
                result = recon_node.result()
            except CancellationException as ex:
                # If reconciliation task was canceled, we were probably canceled too, but
                # that's immaterial as we ultimately have failed for the same reason.
                raise CancellationException("Reconciliation task was canceled.") from ex
            # Let any other exception bubble up as-is.

            return result

        # Build a callable whose signature exactly mirrors the wrapped agent's args,
        # while enforcing keyword-only params after RunContext to satisfy CodeFunction contract.
        names = [a.name for a in agent.args]
        dict_kv = ", ".join([f"'{n}': {n}" for n in names])
        if names:
            params_sig = "*, " + ", ".join(names)
            src = (
                f"def _ensemble_call(ctx, {params_sig}):\n"
                f"    args = {{{dict_kv}}}\n"
            )
        else:
            src = (
                f"def _ensemble_call(ctx):\n"
                f"    args = {{}}\n"
            )
        src += "    return __impl(ctx, args)\n"
        ns: Dict[str, Any] = {"__impl": __impl}
        exec(src, ns)

        super().__init__(
            name=fn_name,
            desc=agent.desc,
            args=list(agent.args),
            callable=ns["_ensemble_call"],
            uses=[agent, self._reconcile_agent],
        )

reconciliation_prompt: str = """
# Ensemble-of-Answers Reconciliation Task

## Objective

You have just completed extensive internal deliberation on the query above through multiple independent reasoning chains. The "Food for thought" sections above represent your own thorough exploration of different approaches, perspectives, and analytical angles you've already worked through internally.

Now, you need to synthesize all of this prior thinking into your final response. Draw upon the full depth of analysis you've already conducted, but present your answer as a cohesive, well-reasoned response that naturally incorporates the full breadth and depth of analysis you've already completed, without referencing the individual reasoning chains as separate entities.

Stated more fully, your task is to:

- Internalize and draw upon all the insights, ideas, approaches, and considerations from your prior analysis. Do not reference "the first approach" or say "another consideration was" or similar meta-commentary about your thinking process. Instead, present your synthesized conclusion as your direct, authoritative, well-reasoned answer that naturally incorporates the depth and breadth of analysis you've already completed.
- Identify the union of strongest reasoning and most compelling evidence across all your thinking.
- Resolve and reconcile any contradictions or tensions that are now apparent.
- Present a final answer that reflects the full depth and breadth of your deliberations.
- Synthesize if you've just completed very thorough internal reasoning as parallel sub-tasks and are now ready to give your best, most considered response, which is the only thing that the user will see.

## Examples of Synthesis (for illustration of what we mean by "Reconciliation Task"):

**Architecture Discussions**: Combine the breadth of alternatives considered with the depth of analysis. For example, if one reasoning chain identified why a particular Design Pattern won't work, while another missed that critical flaw but focused on why it would be effective, you might:
- Incorporate both insights naturally noting the limitation.
- Elicit further thought on how the limitation could be overcome, or another response may already contain good ideas of how that limitation is overcome.
- Reconcile differences of how the design pattern is used in the alternative answers, and only keep the particular way that doesn't have the limitation identified. If one reasoning chain incorporates a particular design pattern much better than another, just use the better way if it's clearly superior.
If the answers bring up different design patterns that are well-justified alternatives, you may want to synthesize these alternatives into the response if they have merit.

**Implementation Tasks**: This could be, for exampe, authoring a class or function that is part of a larger design. Synthesize an implementation incorporating the best elements from different approaches, accounting for edge cases and requirements that some solutions address more thoroughly than others, or simple goofs that some make while others don't. You also want to lean toward whatever is closer to the design spec and plan, to increase alignment and adherence.

**Validation Scenarios**: This could be review of a design spec or an excerpt of code implementation. Since we often do scientific or financial computing, design or code review also covers methodology and correct practice of stats, ML, and math, not just software architecture and code. If one reasoning chain identified incorrect methodology, an edge case, or potential bug that other answers overlooked, incorporate that analysis. Conversely, if one incorrectly identified a bug that others correctly and proactively showed was handled properly, resolve that contradiction.

**Documentation/specification work**: Take the union of correct ideas, but without redundancy, using the clearest explanations and most comprehensive coverage achieved across all your thinking. What key aspects, pertinent details, or core concepts did one alternative cover that others missed -- incorporate all!

## Outliers: Brilliance or Hallucination?

When only one or two of the prior answers have a unique finding or idea, there are always these possibilities:

1. The finding is a hallucination, factually incorrect, based on flawed analysis, or a very bad idea. This can happen because of incorrect reasoning, model confusion due to contradictions in the context, or "context rot" (the context becomes too extended and the session is too lengthy so the model can't focus).
2. The finding is a spark of brilliance, and represents a sample on the far right tail compared to the sampling distribution of answers. Reaching for astute insights, hard-to-find bugs, and pushing the intelligence frontier is the whole point of why we are doing Ensemble-of-Answers now.

### How to handle:

It is much easier for both humans and LLMs to verify the correctness or goodness an idea (whether a design will actually work, whether a bug is actually a true bug, etc..) than it is to come up with one. When you detect aberrant claims, review the given justification/evidence and determine if it requires further verification. For example, if only 1 out of 5 answers claims a particular bug, you probably want to verify a trace of the bug, or show by logic that it is actually impossible for the bug to occur.

## Minimize Tool Use

The original task (system and user prompts), on which we are now following up with this reconciliation task, may have had tools for being able to do a good job on the original task. You still have access to these tools now just in case you need them for the reconciliation task, but often you will not need anything other than viewing files just for reconciliation. Use best judgment.

Sometimes our agent/tool publishes output to file and uses indirection in the return text. In these cases, you need to read in the alternative answers (do so in parallel at beginning if this is the case) -- always make sure you can see or have retrieved all the candidates fully before commencing reconciliation.

## Final Output

- These "Food for thought" (alternative answers culminated from your prior reasoning chains) and these special instructions (the Reconciliation Task) will NOT appear ever again and will be cut out of your final deliverable and what is presented to the user.
- Only your next final response is the deliverable.
- Your final response should read like an independent, single deeply-considered (and thus extremely thorough, comprehensive, and lengthy) response that encapsulates all the worthy insights of this exceptionally thorough reasoning ensemble and leverage your complete prior analytical exploration.
"""
