from ..core import CodeFunction, FunctionArg, RunContext, AgentException, AgentNode

class RaiseException(CodeFunction):
    def __init__(self):
        super().__init__(
            name="raise_exception",
            desc="Agent uses this to declare failure and abort the task with a concise reason.",
            args=[
                FunctionArg("msg", str, desc=
                  "Short reason. Include inner cause if relevant (e.g. could not open a given file path).")],
            callable=self._raise_exception_callable,
        )

    def _raise_exception_callable(self, ctx: RunContext, *, msg: str):
        agent_name = "<unknown>"
        node_id = -1

        # Must be invoked by a CodeNode, which in normal cases has a known
        # parent AgentNode, since that is the whole purpose of this.
        assert ctx.node and ctx.node.parent
        assert isinstance(ctx.node.parent, AgentNode)
        parent = ctx.node.parent
        agent_name = parent.agent_fn.name
        node_id = parent.id

        raise AgentException(msg, agent_name, node_id)


# Built-in global singleton for author reference.
raise_exception = RaiseException()
