`netflux` is :

- minimalist python framework for authoring custom agentic applications. The core idea is to treat agents like any code function in imperative programming: you have inputs, outputs, logic composition (by calling on building blocks of other functions), and side effects on structures.
- our goal was to build a framework that is semantically flexible to do the equivalent of both workflows and dynamic, open-ended problem solvers. Fundamentally, we believe hierarchical Task Decomposition is important to doing this effectively, the same cohesion and reusability of libraries and utility functions in traditional programming.
- explain the central abstraction `Function`.
- explain `CodeFunction`. Refer to the `TextEditor` example in func_lib/text_editor.py as a straightforward but very high utility example.
- explain `AgentFunction`. and why the abstraction of treating it like a Function works well. We may often say "Agent" and we mean an instance of `AgentFunction`. Refer to the `ApplyDiffPatch` example in func_lib/apply_diff.py as an example of a straightforward high utility example. The story with this particular example:
    - it is that often we want agents to be very focused, such as editing an implementation for a new feature or bug fix (planned in an earlier step), without burdening that same agent of being responsible for making the file edits. Or perhaps we want a human to inspect the diffs before we proceed. Thus, we often want to separate patch creation from patch application. `ApplyDiffPatch` is specialized in, once you have the patch, applying to successfully especially in the case where there can be fuzzy match due to imperfections often in whitespace matching. (please state this much more concisely than I did!).
- explain how any Function can call any Function, e.g. agents can invoke code, other agents, and code can invoke agents. Code invoking code is classic programming, and this framework attempts to enable the other 3 of 4 combinations.
- The framework is both the convention / pattern for specifying such functions (defining the building blocks of agents for your application) but also a lightweight execution infrastructure for running, monitoring, debugging, tracing agent instances.
- the concept of **Task Decomposition**: this is an important design philosophy: just as we build on a foundation of cohesive functions for writing higher layer application logic, and such is observable in a call stack, the idea is to do exactly the same when we are defining agents as `AgentFunction`s: a high-level agent should build on a foundation of more specialized sub-`Functions` (in the case of sub-agents, this is just modeled as an `AgentFunction` that invokes another `AgentFunction` to do one particular sub-tasl). So just as good programming practice is to do functional decomposition, we attempt to encourage the same with this framework.
    - this becomes especially important for agents because the importance of focused tasks with limited context and deliberate context window is currently the bottleneck to the successful applications of LLMs for agentic work.
- An agent can break up its work into sub-tasks, and may delegate a sub-task to another agent that is well-suited to executing that sub-task. However, circular references must be impossible. Recursion should also be disallowed to ensure no runaway scenarios.
- When agents invoke "tools" we semantically are referring to physically the act of an agent invoking tools. However, in our framework this is just the physical mechanism by which an LLM is invoking what we actually consider to be functions. 
- An agent is invoked as a Function invocation of an AgentFunction. Every agent has a schematized function that is used to invoke it. When a developer specifies that one agent is able to call another agent to complete a sub-task, this will be translated to making the latter agent's function entrypoint as a tool/function available to the former agent.
- What exactly is a CodeFunction then? Many people consider "tools" to be what we would actually call leaf `CodeFunction`s which are functions that do not invoke any other `Function`. Examples of this are viewing a file, replacing a string in a file, running bash commands. These are the most basic building blocks. Examples of higher-level `CodeFunction`s:
    - a workflow with fixed logic (thus easily described in deterministic code) that coordinates fan-out of work to other `AgentFunction`s. A `CodeFunction` could invoke a `CodeFunction` through the `Runtime`, however usually you would just have a function call instead. Ordinarily, a `CodeFunction` will invoke one or more `AgentFunction`s.
    - a wrapper around an `AgentFunction` to decorate or enhance it. The best example of this is something like the `Ensemble` utility (`func_libs/ensemble.py`), which ...
- We model a function invocation as a "task". We implement the running of a function invocation using a `Node`. Thus, we often refer to a function invocation also as a "task" or as a `Node`. An `AgentNode` tracks the state of a running `AgentFunction` call, including past `Function` calls it has in turn made (a sequence of children `Node`s), a transcript of the full LLM session of the current instance as it progresses (for tracing and observability), and manages the "agent loop" which uses a provider's SDK against their remote endpoint. Importantly, a `Node` represents the state and history of a function call both while it is executing and after it is done executing.
- Tree of `Node`s is kept around even after a top-level task is complete, until the user deletes it, for post-run debuggability and traceability.
- Explain that external consumers don't use Node directly (except for CodeFunction authors), they instead use `NodeView`. Consumers use `NodeView` to ensure that they are seeing atomic checkpoints of a tree's total state. Discuss more about `NodeView` and why it is necessary and the idea that it is a single consistent view. Show an example with the `watch()` loop of how it is supposed to be used, briefly.
- At any point in time, if you take a snapshot of the call hierarchy, this looks just like a traditional call stack, except now a Function can be an agent doing a task.
- In Object-oriented programming, methods are just functions that can mutate state of an object. To support this, we introduce the concept of a `SessionBag` which is a way for objects to have a lifetime of a task. 
- The execution of a task, including its subtasks, can be modeled as a tree where each task is a Node and each Node's children are an ordered sequence of edges to children Nodes representing the order in which the function called other functions.
    - In asynchronous programming, concurrent function calls can be done using Futures/Tasks. We also permit a similar model by having a caller not need to invoke blocking Node.result() until it is ready to wait -- similar to a Future. Thus a Function is free to invoke other Functions in parallel. For an AgentFunction, this is supported for LLMs that have "parallel tool calling" as a feature.
- A "top-level task" is a Function call where the caller is external to the framework and agents spec. For example, when a user app or web server consumes the framework (and a Functions spec on which it is instantiated) to invoke a top-level Function, which could be either a `CodeFunction` or `AgentFunction` alike, this is a top-level task instance.
- An agent may be invoked as a top-level task externally or as a sub-task (invoked by another AgentFunction or CodeFunction). The external caller may choose to invoke a broad-scoped AgentFunction that in turn coordinates a complex task, or the external caller may directly invoke a much more purpose-specific AgentFunction that might also be used multiple layers deep within some other broader AgentFunction -- there is no need to force an agent as always being one or the other. There is no concept of top vs non-top Functions: any Function can be either, just as in normal programming.
- A long-running workflow-like agent will usually play more of an orchestrator agent kind of role, breaking the problem into sub-tasks which may change dynamically as progress is made.
- Function Call Stack Analogy: as you go deeper in a call stack trace, functions are more specialized, until you get to the lowest-level library functions. In our Call Stack, function calls are instead agents (AgentFunction) or traditional code functions (CodeFunction), where usually at the bottom of the stack one will find CodeFunctions utilities that operate like built-in low level library functions. Agents are like higher level functions, and at any point in time during execution you can take a top-level task and visualize the netflux Call Stack like we do a traditional function call stack.
- **The logical reasoning of an agent replaces the fixed code logic of a function, but other than this we see little difference between traditional functions and agents, which is why we base the framework on this fundamental idea**.
- A highly specialized agent will often have only leaf tools (for example, if it needs to analyze inputs that are files or produce output to files), or no tools (for example if it is only doing analysis), and is going to be deeper in depth of the netflux call stack at any time.
- Every agent is specified as:
    - schematized invocation arguments such that the agent can be invoked as a function call
    - system prompt with optional string substitution using one or more of calling function's arguments.
    - 1 initial user turn prompt - ditto.
    - Specifics of the task (agent invocation) are injected into the system and/or user prompt in a predefined way - typically this would just be string substitution but it can also be whatever the author desires as long as the transformation of fixed arguments -> concretized prompt can be specified.
    - Short description of the agent's purpose and arguments.
    - List of other `Function`s (defined in this framework) that can be invoked by the agent. That can be for leaf tasks like file editing or sub-task delegation.
    - Agents can be given the `RaiseException` function which will allow them to raise an `AgentException` for any reason.
- Give a briefer on the Exception Model that we use. The key concept is that we make `Exception`s fluid with `Function`s just like in traditional programming. Explain why an agent would want to raise an `Exception`. Refer to the `Exception Model` section for more details (give hash link).
- Providers are subtypes of AgentNode that bridge the framework's pattern to the model's SDK. refer to section below for how to write a provider extension for a new model. Give a brief idea of how a new `AgentNode` provider would be added and then refer to the relevant section below. 
- Agent instance token accounting. Discuss in a couple sentence what we track. Refer to relevant section below.


Tips & Tricks

- The framework tries to make it easy to do effective context engineering. Usually higher-level agents will have the role of orchestration or workflow. Lower-level agents will solve more concrete patterns of problem. As LLMs become more sophisticated, a single agent can take on a broader set of responsibilities (more Functions as its disposal, and longer-running). Partition sub-tasks as fine-grained as needed but don't over-partition unnecessarily as this can degrade your evals.
- agents may use files for input(s)/output(s). Input filepaths would be given as args, and output filepaths may be returned (Write File tool used prior to returning).
- structured outputs are discouraged since LLMs are sophisticated enough to parse unstructured outputs from their sub-tasks. However, sometimes strict structured outputs are critical, and this can be enforced by defining `CodeFunction`s where the arguments are the schema and the Callable performs serialization and/or verification, depending on the reason for the structured output, and returns empty or provides a filepath with the serialized data, etc. You can leverage the framework's Exceptions Model to propagate an Exception if verification fails.
- When authoring agents, place the common prompt before the specifics of the task instance. This is because LLMs are known to pay greater attention to the beginning and end of the context window. Particularly, when giving background information, whatever most heavily will influence the specific actions the agent will take should be placed closer to the end of the prompt.
- human_in_loop()
    - becomes blocking for human input. Human can interject and this content will present forward guidance in the "function" output.
    - implement the hook to human UI using whatever mechanism particular to your application.
    - various reasons why model may choose to invoke: (a) sign-off at key points; (b) lacking confidence and need guidance on the task; (c) on the verge of RaiseException and seeking opinion of what to try before doing so.
- may help to be slightly repetitive of system prompt elements in user prompt to get better adherence.
- system prompts kept tiny and stable: agent’s role declaration, non‑negotiable rules/guardrails, output contract, meticulosity, verbosity/brevity, tool‑use policy (steer how often and when to use certain tools, beyond tool schema).
    - "you focus on performance optimization of the algorithm already select; do not propose new algorithms, just optimize impl using the one chosen."
    - "you must use tools to test performance and confirm speedups. You cannot just be speculative -- your results need to be backed up by numbers and you can admit lack of improvement."
- user prompt: (1) all the agent-specific context of the generic problem background (even if common to all instances this is still not system prompt), (2) the specific problem instance the agent is being invoked to do now.


-----


# Entities & Architecture

## `Function`

`Function` is the central abstraction whereby code or agents are both abstracted as merely being kinds of function calls. Specification / metadata describing the agent or code.

* Concrete subtypes must override abstract property `uses() -> List[Function]`, specifying any `Function`s that can be `invoke()`d by the `Function`.
* `AgentFunction`: can be invoked by any `Function`.
    * user subtypes to define their own agents (could use abstract properties that user must override).
    * subtype must specify: input vars, system prompt, templated user prompt (var substitution). Each var may be given as strings or filepaths (upon instantiation of the agent, files would have to be loaded and then substitution done by the runner infra instead of asking the agent to do it).
    * subtype specifies `uses: List[Function]` — the `Function`s that the agent may invoke.
* `CodeFunction`: can also be invoked by any `Function`.
    * some framework built-in subtypes (`Ensemble`, `ThinkMoreDecorator`).
    * mostly user subtypes to define any plain python functions that do some deterministic logic, intended to be invoked most often by `AgentFunction`s or as the top-level request, to coordinate sub-agents doing sub-tasks.
    * may also invoke another `CodeFunction` within their code although this will be less common.
    * points to a python function Callable. First arg is a `RunContext` which is used to invoke the framework to run a `Function`.
    * spec gives the arguments (names, types, description) without the `RunContext`. Framework will later check that the Callable matches the spec + the `RunContext` arg present. For now only allow basic primitive types (string, int, float, bool). Use python primitives to indicate types.
    * in user python code (inside the Callable), user can invoke other `Function` by doing this:
        * invoke another `CodeFunction` via:
            * Just call the callable directly (regular python code calling a function); framework does not see this happening and it's perfectly allowed.
                * in the case of `Ensemble`, user could theoretically use: `<Ensemble instance>.callable` after they have one.
                * Pass the same `RunContext` through.
                * No need to include the invoked function in the `uses()` property.
            * Use the framework runner infra to invoke, via `RunContext.invoke()`. Possible invocation of a `CodeFunction` must be declared in `uses`.
        * invoke an `AgentFunction`:
            * use the framework runner infra to invoke, via `RunContext.invoke()`. Possible invocation of an `AgentFunction` must be declared in `uses`.

## `Runtime`

`Runtime` is the top-level runner responsible for execution of trees and managing their state.

* Framework-provided object encapsulating all runner infra
* Created with a collection of user-defined `Function`s (hierarchy) that may be invoked directly or indirectly; Author defines `Function`s fully before creating a `Runtime`.
    * `runtime = Runtime(specs: List[Function])`
    * `runtime.get_ctx() -> RunContext`: return a special `RunContext` that is outside the scope of any Task (`Function` invocation).
* During registration, the runtime automatically performs a **BFS over each Function's `uses` graph** to discover and register all transitively referenced Functions. Consumers may seed with a partial set; transitives are added automatically. Duplicate names that point to different Function instances are rejected.
* Responsible for creating trees of `Node`s that execute `Function`s.
    * `RunContext.invoke()` posts `Function` invocations to the `Runtime`.
    * `Runtime` creates child `Node` for the invocation, updates the relationship in the `Node` caller, and updates its own `Node`-indexing data structures.
    * `Runtime` drives the child `Node` to start when resources are available (i.e. agent concurrency control is managed by the runtime).
        * `CodeNode` will always be started immediately.
* Provides consumer interface for querying trees of `Node`s.
    * e.g. visualization
    * `list_toplevel_views() -> List[NodeView]`: get consistent snapshot of all root tasks
    * `get_view(node_id: int) -> NodeView`: get latest snapshot for any node without blocking
    * `watch(node: Node | int, as_of_seq: int = 0) -> NodeView`: block until newer snapshot available (read more about `NodeView`s below).
    * To prevent race conditions, consumers should use `Runtime` to query state via `NodeView`s and do top-level invocations.

## `RunContext`

`RunContext` is a common framework interface used by both framework consumers and framework internal impl to invoke `Function`s.

* serves at least as the interface for:
    1. top-level task invocation; called by an app that is consuming the framework and a collection of `Function`s (the app or someone else may define these); access via `Runtime.get_ctx()`.
    2. python code for user-defined or framework-builtin `CodeFunction`s that invoke other `Function`s.
    3. when some framework component needs to handle agents doing tool calls, that component delegates invocation to the `RunContext`.
    * e.g. they all use: `ctx.invoke(fn: Function, args: Dict[str, Any], provider: Optional[Provider] = None) -> Node`
* every `Function` invocation has a `RunContext` given to it, providing the interface, but also tracking the particular `Function` using it.
    * when a `Function` invokes another `Function` (including when framework handles `AgentFunction` invoking any `Function` via tool call), the `RunContext` knows its associated invoking `Node` (identity of the caller) and causes creation of the invoked `Node`.
        * this information is used to construct the directed edges relationships of the `Node` tree. A single top-level task invocation is the parent `Node` of a tree.
* Each top-level `ctx.invoke()` (by consuming app) initiates one tree of `Node`s where the parent `Node` of the tree is the top-level Task.
    * Each top-level Task is an independent tree with `Node`s disjoint from those originating from other top-level tasks.
    * Each top-level Task may originate from the invocation of **any** `Function` that was registered with the `Runtime` constructor, thus being coarse-grained tasks or fine-grained tasks at the top level.
        * General idea: Fine-grained top-level Tasks would appear as shallow trees, that may be comparable to the deepest subtrees of a coarse-grained top-level Task that decomposes into the former -- the latter being a broader-scope task that needs to solve the former's scope of problem perhaps as a mere sub-sub-Task.
* Fields:
    * `node: Optional[Node]`: a reference to the particular `Node` identifying this specific `Function` invocation. `None` for top-level contexts.
    * `runtime: Runtime`: a reference to the shared `Runtime`.
    * `object_bags: Dict[SessionScope, SessionBag]`: references to session bags accessible at different scopes.
* Methods:
    * `invoke(fn: Function, args: Dict[str, Any], provider: Optional[Provider] = None) -> Node`: invoke a `Function` and return the created `Node`.
    * `post_status_update(state: NodeState)`: update the current node's status.
    * `post_success(outputs: Any)`: mark the current node as successful with given outputs.
    * `post_exception(exception: Exception)`: mark the current node as failed with given exception.
* Narrow Scope: `RunContext` is just a mechanism to pass on `Function` invocation directives to the `Runtime` to act on them.

## `Node`

`Node` is an core abstract object that represents the invocation of a `Function` (which we also call a "Task").

* `AgentNode`: represents and manages the state and running of an `AgentFunction` invocation.
    * `AnthropicAgentNode`
        * particular implementation when the `AgentFunction` is invoked with Anthropic LLM (e.g. Opus 4.1).
    * `GeminiAgentNode`
        * particular implementation when the `AgentFunction` is invoked with Gemini LLM (e.g. Gemini Pro 2.5).
    * Tracks history of LLM session thus far (which it also uses in tool cycle when doing follow-up request)
        * Subtypes `AnthropicAgentNode` and `GeminiAgentNode` store and use the SDK-specific types in their internal impl.
    * `node.get_transcript() -> List[TranscriptPart]`
        * Subtypes must implement; they must convert the SDK-specific types in the transcription they are tracking to the framework-common `TranscriptPart`s. They never convert types in the reverse direction.
    * Child `Function` invocations are tracked in `node.children: List[Node]` property.
        * Always ordered to reflect the sequence in which `Function`s were invoked. 
        * For consumers outside the framework, use `NodeView.children: tuple[NodeView, ...]` instead to access child information safely.
    * Has states (Waiting, Running, Success, Error) but also sub-state including tool use (`Function` invocation) that it is waiting on.
    * `AgentNode` is completed once it returns final assistant text or the model decides to `RaiseException` (if it has been given as an option).
    * `TokenUsage` cumulative accounting must be reportable by every `AgentNode` and kept up to date throughout the agent loop (updated on every request/response iteration).
        * Subtype implementations must use the provider SDK's token usage meta to track the accumulation.
* `CodeNode`: represents and manages the state and running of a `CodeFunction` invocation.
    * Simpler than `AgentFunction` because it is just a function call (unlike LLM session complexity). Authors invoke `Function`s directly from within the `Callable`.
    * `CodeNode` is completed once it either returns or raises.
* Fields / Properties:
    * `id: int`: monotonically increasing unique identifier when Node is to be used as key in any lookup. This is one and the same as "task id".
    * `fn: Function`: which `Function` the `Node` is an instance of.
    * `inputs: Dict[str, Any]`: What the inputs were for the invocation.
    * `outputs: Optional[Any]`: What the output(s) were from the run (if finished). Usually just an unstructured string.
    * `exception: Optional[Exception]`: the exception, if there was an exception.
    * `state: NodeState`: (Waiting, Running, Success, Error) enum
    * `children: List[Node]`: ordered list of child `Function` invocations made by this `Node`.
    * **Note**: External consumers should access this information through `NodeView` instead of `Node` directly to avoid race conditions.

## `TranscriptPart`

`TranscriptPart` represents each of the parts that are common to the model-specific SDKs in concept. The subtypes:

* `UserTextPart`
* `ModelTextPart`
* `ToolUsePart`
* `ToolResultPart`
* `ThinkingBlockPart`
    * including both redacted and non-redacted
    * includes `signature` field for thinking block signatures
* On every follow-up call, replay the full history in original order.

## `Ensemble`

* Is a `CodeFunction` that decorates any `AgentFunction` to do parallel independent invocations followed by reconciliation.
    * First phase: each `AgentFunction` call proceeds as normal, with args forwarded for normal user prompt substitution.
    * Second phase: same system prompt and substituted user prompt; append to the user prompt each of the completions along with a reconciliation instruction.
* Given any `AgentFunction`, mostly users will construct one from built-in factory facility (ctor directly):
    * `Ensemble(agent: AgentFunction, instances: Dict[Provider, int], name: Optional[str] = None, reconcile_by: Optional[Provider] = None)`
        * `instances`: how many parallel invocations of `AgentFunction` to do with each model.
    * User uses this when defining their `Function`s.
* Automatically has a valid inner Callable like any `CodeFunction` that does the ensembling phases.

## `SessionBag`

* Collection of arbitrary objects that may be read, mutated, and persisted by `Function`s.
* Each `Node` created introduces a `SessionBag` with its lifetime. The `Node` and its children can access the bag.
    * Thus, the `Node` can also access its parent's `SessionBag`, if it has a parent.
* Each `Node` can also access the `SessionBag` of the root `Node`.
* `SessionScope`: enum of lifetime scopes, each of which would refer to a different `SessionBag` that a `Node` can access:
    * `TopLevel`: Lifetime envelopes all Nodes in a top-level tree. This would give the root `Node`'s bag.
    * `Parent`: Lifetime of the `Node`'s parent `Node`. This would give the parent's bag.
    * `Self`: Lifetime of the `Node` itself. This gives the `Node` access to its own bag.
        * The main application of this is for a `Function` to receive results from its children and to act as a scratchpad.
* Mechanism to do object-oriented programming
    * `Function` operates on an object and thus can behave like a method.
    * `Function` can accept arguments that refer to objects; pass data between `Node`s by in-memory strong types instead of requiring ser/des or free-form text.
* Mechanism for `Function` to own its own objects and invoke `Function`s that read/create/mutate them.
    * Example: an `AgentFunction` needs its own persistent Bash session (e.g. process tree, env vars, vars, cwd)
        * To be used at random points over its lifetime
        * Example: launch executables asynchronously (in terminal background; running locally on client); retrieve results later after doing other steps.
* `RunContext` of a `Node` carries the references to the three scopes of `SessionBag`s.
    * For a root `Node`, the `TopLevel` bag is the same as the `Self` bag. Trying to access the non-existent `Parent` bag raises `NoParentSessionError`.
    * For children of the root, the `Parent` bag is the same as the `TopLevel` bag.
    * Any deeper `Node`s will find the bags of the three `SessionScope`s to be different.
    * `RunContext.get_or_put(scope: SessionScope, namespace: str, key: str, factory: Callable[[], Any]) -> Any`
        * To simplify, this is the only mechanism to be used by `Function`s for access. Concurrency-safe in case of parallel `Function` invocations. Simplify by invoking `factory` under the lock since not high-frequency.
        * `Function` implementations should cooperate to use descript namespaces and keys, composed of static string constants and instance numbers if multiplicity is possible.
    * `Runtime` is responsible for creating `SessionBag` with each `Node` and propagating references to new descendants.
* Framework will currently rely on ref counting, garbage collection, and self-disposing object behavior (author responsibility).
    * `Runtime` destruction, or explicit user request to delete a finished tree, will induce disposal of all `SessionBag`-referenced objects and their resources.
    * This keeps objects alive long past their usable scope (potential resource leak), but is very worth the debuggability for finished subtrees. We can make this more configurable in the future (e.g. mandatory finalizers and dispose on `Node` completion).

## Exception Model

* Any `Function` can raise or bubble up an `Exception` at any point while running.
    * For `CodeFunction`, this is just for the vanilla reasons:
        * `raise TException(..)` in the `Callable`, e.g. due to contract breakage, bad args, business logic, assertion failure, etc.
        * Bubble-up: its `Callable` invokes a regular function that in turn raises and the `Callable` is unable to handle it or recover.
    * For `AgentFunction`, this is the agent making a proactive intelligent decision that it wants to raise an `Exception`.
        * All the reasons in classical programming, but also:
            * Agent is unable to do as directed because:
                * lacks context or key knowledge
                * lacks the sub-`Function`s it needs (leaf tools or sub-agents) due to author error
                * sub-agent (invoked `AgentFunction`) is not behaving as expected on a sub-task
                * an invoked child `Function` has raised, and it's unclear how to handle or it's recurring, and there are no alternatives or the alternatives have already been tried.
        * An agent may be given guidance on:
            * when and which `Exception`s from children `Function`s to recover from, versus when to bubble them up.
            * when to decide the given task is unsolvable and give up by raising.
        * Encourage the agent to declare failure to reduce the rate of hallucination.

### Framework support mechanisms:

* Framework built-in `RaiseException` (is-a `Function` subtype) intended to be provided to agents in their `AgentFunction.uses` definition, by voluntary opt-in from the agent author.
    * The spec instructs usage directives like:
        * Bubbling up an `Exception` it can't solve? Include an inner exception type and inner msg inside of the `msg` arg.
        * No alternatives worked? Very briefly describe what was tried.
        * Missing information or context, don't know how to solve, etc? Describe this very briefly for the caller in case a follow-up attempt could address this.
    * Implementation of the `RaiseException` callable is a one-liner: raise the `AgentException`. (Assume `CodeFunction` never invokes it).
* Differentiation of agent vs. service/infra faults:
    * `AgentException`: used when an agent decides to invoke `raise_exception(msg)` by its own volition, for any reason.
        * includes: faulting agent's name and instance id (`Node.id`).
    * `ModelProviderException`: used when an `AgentNode` implementation (`providers/`) fails for any reason.
        * Always unrelated to the agent's task and never caused by an agent.
        * includes: provider class name, name of agent being processed when provider faulted, instance id (`Node.id`), inner exception object.
        * Examples:
            * provider `AgentNode` malimplementation (not following protocol; not using SDK correctly)
            * connection socket broken or can't open
            * authentication / configuration
            * provider is overloaded, client is being rate-limited, any kind of load shedding or quota issue
            * core framework bug (developer regression) e.g. during `ctx.invoke(..)`.
            * other faults by the remote provider service
* Accessing `Node.result()` will either return the `Function` output (usually a string) if it was successful, or will raise the `Exception` from that invocation if there is one (similar to Futures in many languages), regardless of the invoked `Function`'s subtype.
* Provider-specific `AgentNode` subtypes shall implement this contract:
    * When collecting `Function` invocation results from `ctx.invoke(..).result()`, expect the possibility of an `Exception` being raised and always catch it.
        * Pass on a string representation of the `Exception`'s type and message (with details but never too verbose and never with stacktrace) back to the LLM in the regular follow-up tool cycle, and flag the fault if the provider's SDK has an explicit field for that. Some LLMs are fine-tuned to pay attention to the error flag but most will understand the `Exception` string properly anyway especially if the detail is present.
        * Includes `ValueError` for built-in argument type checking (LLM can respond by re-trying).
    * Implement backoff-retry around SDK `Exception`s that are known to be transient only.
    * Intercept `RaiseException` calls and use `ctx.post_exception(e)` where `e` is an instance of `AgentException`. Then exit the run loop.
    * Allow any other unexpected `Exception` to bubble past `run()`. The supertype `AgentNode` will wrap it in a `ModelProviderException` with context.
    * A batch of parallel tool calls may result in 0, 1, or more of them succeeding or excepting and this is normal.
    * When a model issues a batch of tool calls and one of them is RaiseException (unusual), honor the model's intent and propagate `AgentException` to end the agent loop after the whole batch is attempted.
* `CodeFunction` authors guidance:
    * Be aware that `Node.result()` from invoked functions may raise.
    * Ensure raisable `Exception`s from the Callable have descript type names and sufficient detail. If bubbling, sometimes this requires try-catch interception just to augment details (e.g. is the error pertaining to an input or output) and then re-raising.
    * Consider `AgentException`: may be difficult to handle statically; consider: retry, change provider. If repeatable, wrap attempts, augment context, and bubble up.
    * Consider `ModelProviderException`: Avoid backoff-retry to prevent multi-layer retry. Log in case of bug. Augment context and re-raise.
* `AgentFunction` authors guidance:
    * Add the built-in `RaiseException` to the `AgentFunction.uses` property to enlist in `AgentException`s.
    * Provide additional guidance on when to raise, when to bubble up, (or how hard to retry alternatives first) in the system and user prompt. Iterate through trial and error. This will be very specific to the agent's purpose and scope.
    * Strongly consider instructing the model to invoke `human_in_loop()` **before** considering `raise_exception()`.

## `NodeView`

`NodeView` is an immutable, consistent snapshot of a `Node` and, through reference, its entire subtree, intended for external consumers to observe task tree state without races. Do not read `Node` fields directly from UI/visualizers or outside code running inside a `CodeFunction` — those mutate concurrently while tasks run and are part of the framework's object model. Instead, use `NodeView`s.

- Purpose: provide a race-free, immutable view of a (sub-)tree at a single global version. The snapshot includes most fields in `Node` (e.g. `state`, `outputs`, `exception`, `children`).
- Consistency model: the `Runtime` maintains a global sequence number. On any change (node creation, status updates, success/exception), it rebuilds the `NodeView` for the changed node and all ancestors and updates the latest cached `NodeView` tree as of that version. Notice that siblings and siblings of the ancestors don't need to have their `NodeView`s regenerated because they aren't affected. Consumers can only acquire `NodeView` trees in-between these updates. Each `NodeView` also has `update_seqnum` giving when it was last updated.
- Consumer Watcher loop: call `node.watch(as_of_seq=prev_seq)` to block until there is a newer snapshot (where `touch_seqno > as_of_seq`) and then receive the latest `NodeView` for that subtree. Start with `prev_seq = 0` and after each update set `prev_seq = view.update_seqnum` to continue receiving atomic updates. This is similar to etcd/zookeeper watchers, except the `Runtime` keeps only the latest view available. Watcher loops should be used for event-driven UI.
- Top-level views: use `runtime.list_toplevel_views()` to get a consistent snapshot of all root tasks at once; `runtime.get_view(node_id)` returns the latest snapshot for any node without blocking.
- Why: this isolates observers from partial, in-flight mutations during execution and guarantees each delivered view is a self-consistent global snapshot of the tree at a specific sequence number.

## Deferred Features

This is a bucket list of nice-to-haves.

* Concurrency control
    * Limit the number of `AgentNode`s in the agent loop concurrently, keyed by `Provider`.
* Replayability, Pausability, Interruptibility, Cancelation
    * Pre-requisites:
        * Restart-ability of tree from the state where any `Node` was just created.
        * Serializability of `Node` tree state.
        * Cancelation, Pause `NodeState`.
* `NodeState.WaitingOnFunction`
* Async and Futures
    * `RunContext.invoke()` -> return Future and generally use async chaining.
* Streaming SDK usage in model provider `AgentNode` run loops.
    * For observability, debuggability.
    * Compress results into the SDK's full native type blocks after each block is done streaming.
* Fully migrate to Event-Driven Architecture.
    * Single event loop per Runtime; remove the per-`Node` thread.
* Smarter caching based on past `Function` instance statistics.
* Cycle & Recursion Prohibition
    * Enforce at `Runtime` construction.
    * Ensure each `Function` has legal references to other `Function`s.
    * Reject if not a DAG.
    * Reject if function type annotations do not match the spec in `CodeFunction`.
