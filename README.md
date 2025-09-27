(`README.md`)

- `skynet` is a python library we are designing and implementing for custom long-task / long-workflow agent development.

Concepts:

- Framework is generic. Developer uses the framework to specify a set of task-specialized agents. The framework is the convention for this specification, the patterns (e.g. agents delegating sub-tasks to agents as tool calls), and a lightweight execution infrastructure for running, monitoring, debugging, tracing agent instances.
- An agent can break up its work into sub-tasks, and may delegate a sub-task to another agent that is well-suited to executing that sub-task. However, circular references must be impossible. Recursion should also be disallowed to ensure no runaway scenarios.
- An agent is invoked as a tool call. Every agent has a schematized function that is used to invoke it. When a developer specifies that one agent is able to call another agent to complete a sub-task, this will be translated to making the latter agent's function entrypoint as a tool call available to the former agent.
- The abstraction of a tool will be broken into "Agent Tool" and "Leaf Tool". Leaf tool is something like file edit, bash command, or searching for information -- non-agentic tools that will not branch out work to more agents.
- A top-level task is the origination agent call external to the framework and agents spec. For example, when a user app or web server consumes the framework library (and an agents spec) to invoke an agent, this is a top-level task instance.
- task == invocation of an agent == session (LLM session). The semantics apply regardless whether we mean top-level task or sub-task.
- An agent may be invoked as a top-level task externally or as a sub-task (invoked by another agent) -- there is no need to force an agent as always being one or the other.
- A long-running workflow-like agent will usually play more of an orchestrator agent kind of role, breaking the problem into sub-tasks which may change dynamically as progress is made.
- Function Call Stack Analogy: as you go deeper in a call stack trace, functions are more specialized, until you get to the lowest-level library functions. In the Agent Call Stack, function calls are instead agent tool or leaf tool calls, where leaf tools are more like built-in low level library functions, and agents are like higher level functions, and at any point in time during execution you can take a top-level task and visualize the Agent Call Stack like we do a function call stack. The logical reasoning of an agent replaces the fixed code logic of a function.
- A highly specialized agent will often have only leaf tools, or no tools, and is the deepest depth of the agent call stack at any time.
- Every agent is specified as:
    - schematized invocation arguments such that the agent can be invoked as a function call
    - system prompt with optional string substitution using one or more of calling function's arguments.
    - 1 initial user turn prompt - ditto.
    - Specifics of the task (agent invocation) are injected into the system and/or user prompt in a predefined way.
    - Short description of the agent's purpose and arguments.
    - List of other agents (defined in this framework) that can be invoked by this agent via tool call. For sub-task delegation.
    - List of custom functions that the developer may bring themselves as leaf tool calls.
    - Model-specific leaf tool calls that are SDK-supported and the model has been optimized to use during its training, or custom functions that the developer brings that should only be used by certain models for any reason.
        - Provided as Dict[ModelEnum: List[?]]
- common convention: agents may use files for input(s)/output(s). Input filepaths would be given as args, and output filepaths may be returned (Write File tool used prior to returning).
- common convention: agents may return structured outputs via the final tool call technique e.g. define a tool called `return` with output type schema particular to each agent.
- Framework will have an object model for managing the state of agent instances including past tool invocations (and agent instances therein). Upon agent instance completion, the same objects continue to serve monitoring, tracing, debuggability, viz, etc. A single object model serves to represent both past and present state.
- Agent state may be queried read-only recursively (i.e. follow past sub-agent calls) in order to present a visualization tree of the path taken from a top-level agent call. Each node in the tree is a tool call. If the node is an agent instance, it may have children nodes, but a leaf tool call is always a leaf node in the tree. Because there is a specific call stack sequence, each edge from parent to child is given an increasing number which is the sequence in which it was called. Where tools were invoked in parallel (some models support), the edges have tied sequence numbers.
- Currently we will start with, and only have available, Anthropic Opus 4.1 and Gemini 2.5 Pro (both with max thinking budget) with tools. We already have a separate framework for working with openai models so disregard openai models and other model providers. We want our framework object model and patterns to be generic but we need to deal with the specific nuances of these two particular models to start with. We will use `anthropic` and `google-genai` pypi packages.
- Agent instance token accounting. When there are multiple tool calls, we keep the sum of accounting. Cache hit input tokens, Cache creation input tokens (anthropic only?), regular input tokens, reasoning output tokens, non-reasoning output tokens.
- An agent should only ever execute serially without parallelism, even though parallel tool calls may be requested by model -- those are still processed one at a time. Thus, an agent (or any of its sub-agents) can only have zero or one api request in-flight at a time.
- Framework shall use a process-wide global semaphore to limit the number of concurrent requests to any model api in flight. When an agent or tool is invoked, it is given a shared object containing a reference to the semaphore and a boolean `has_sem`, which is inherited from the top-level task. Some non-agentic tools that know they are long-running can courteously give up their lease on the semaphore. For example, the human_in_loop tool should do this. After unwinding, when an agent once again needs to make a request to a model, it would need to re-acquire the semaphore lease if it was given up. By default, we do not give up the semaphore in order to make more effective use of prompt caching (which has ephemeral ttl) and to encourage front-of-line agent throughput when many top-level tasks have been enqueued.
- When calling a remote model endpoint api (at any point in an agent's life), any failure should have 5, 10, 15, 20 second backoff and then fail. At that point, the agent is in paused state and human could query framework for status and request to resume.
    - Specifics of error handling and understanding are model SDK-specific.
- When a tool call function fails (catch exception), the agent waiting on it will be put into paused state and ditto.
- Do not attempt sophisticated middle-of-request error recovery: if failure happens in the middle of streaming, just retry or pause at the point of last success (for example, re-trying from last tool result callback or initial request).
- Every agent should be modeled as a state machine. The overall architecture should be event-driven, with a single event handler created when the framework runner object is created (and owned by that object), which runs in the background. When the user uses the framework api, this creates a request event to the dispatcher. The dispatcher deposits the result into a future, which the caller asynchronously waits on.

Claude Opus 4.1 Specific Notes:
- Prompt caching:
    - Longest prefix cache partial hit wins: you get partial credit for cached prefix and the rest are new tokens which you incrementally pay to cache (if cache watermark added again on latest request).
    - Any agent that has no tools should never enable cache watermarks
    - Any agent that has only leaf tools and no human-in-loop tool should add 5 minute ttl ephemeral cache watermark on initial prompt and on every tool result callback. Only put the cache watermark on the latest.
    - Any agent otherwise: before the agent commences any request, the framework should check the last 5 **completed** invocations of that agent and get the average number of tool calls and average time between tool calls. If more than 1 tool call on average and average time between tool calls is less than 1 hour, add 1 hour ttl ephemeral cache watermark on initial prompt and after each tool result callback, else no cache watermark. This behavior stays consistent once the agent starts. Only put the cache watermark on the latest.
    - The framework is responsible for acting on the logic mentioned here and providing caching as either 'none', '5m', '1hr'.
    - Track cache performance using the `cache_creation_input_tokens` and `cache_read_input_tokens` fields in each query response -- this should be part of agent instance token accounting.
- Extended Thinking: always use the maximum output tokens and thinking budget: `{ max_tokens=32000, thinking={"type": "enabled", "budget_tokens": 80000}}`
    - normally budget_tokens must be set to a value less than max_tokens. However, when using interleaved thinking with tools, as we are, you can exceed this limit as the token limit becomes your entire context window (200k tokens).
    - Extended Thinking Interleaved with Tool Use:
        - Add the beta header interleaved-thinking-2025-05-14 to each api request.
        - always use this. Must use: `tool_choice: {"type": "auto"}`.
        - With the header, Claude is allowed to emit new thinking blocks after tool results, potentially leading to another tool_use in the next assistant turn—something it won’t do in the non‑interleaved mode. E.g.:
            - thinking → tool_use(s) → (user sends tool_result(s)) → thinking → text
            - thinking → tool_use(s) → (user sends tool_result(s)) → thinking → tool_use(s) → (user sends tool_result(s)→ thinking → text
            - etc
        - Since we want agentic task completion end to end, you must always use the header.
        - Refer to: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-passing-thinking-blocks-with-tool-results
        - Refer to: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#tool-use-with-interleaved-thinking
        - To simplify caching and api usage, always request with the full history of thinking blocks, tool use request, tool use response, etc, and for the prompt caching policy we select for the agent, always put the `cache_control` on every request latest msg.
        - While tool results appear as user messages in the API structure, they’re part of a continuous reasoning flow. Preserving thinking blocks maintains this conceptual flow across multiple API calls.
        - With interleaved thinking, Claude can:
            - Reason about the results of a tool call before deciding what to do next
            - Chain multiple tool calls with reasoning steps in between
            - Make more nuanced decisions based on intermediate results
        - "With interleaved thinking, the budget_tokens can exceed the max_tokens parameter, as it represents the total budget across all thinking blocks within one assistant turn."
            - so should we try always using `{ max_tokens=32000, thinking={"type": "enabled", "budget_tokens": 64000}}` instead ?
        - When streaming with tool use, you should wait to receive the full tool use inputs before invoking tool, to simplify.
            - Refer to: https://docs.anthropic.com/en/docs/build-with-claude/streaming#streaming-request-with-tool-use
        - If Opus asks for parallel tools call, remember to aggregate tool results into one message per turn even though we will locally process the tool calls sequentially (session: logically concurrent. in our process: serial/sequential but independent).
- Agents will often make available and provide implementation of tools claude was optimized to use:
    - bash tool
        - Refer to: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool
        - implementation given at link above.
    - text editor tool
        - Refer to: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool
        - author or find robust implementation as one is not provided
- Strict structured outputs
    - via schematized tool call `return_result`
- **Intended Usage:**
    - Conversation shape: One initial user text prompt starts the session. After that, every "user" request is just a `tool_result` only (no user `text` type part), or multiple `tool_result` if the assistant requested parallel tool use. And every non-final `assistant` response contains reasoning (or redacted-reasoning) block and signature, [optional `text` block seen sometimes], `tool_use` block (parallel or single). Final `assistant` response contains reasoning (or redacted-reasoning) block and signature, followed by final `text` block.
    - Signatures will be included from the assistant on reasoning or redacted-reasoning blocks. These must be included when sending the conversation history back in user requests.
    - Model will decrypt redacted reasoning blocks when they are sent back (with signatures). It is only the user that cannot see them.
    - Our replay policy: on every user request (tool use follow-ups), you always replay the full conversation history (all elements) since the initial user text prompt, in exact sequence sent and received, unmodified.
    - Continuous reasoning: When this is done properly (Never alter, reorder, trim, or re-wrap any assistant block. You only append new tool_result blocks to transcript and then send the request), and the tool-reasoning interleaving beta header is enabled, and tool choice "auto" is used, you will attain ** full fluent context since the last user text prompt **: the model's follow-up reasoning step has direct access to the entire chain of prior reasoning and actions in its context window. Thus, you get ** reasoning continuity **: the model will produce new thinking that references earlier thinking you replayed. From the model’s perspective, this behaves like ** one continuous, ever‑expanding assistant turn across many tool cycles incorporating continuous reasoning **.
    - I observed direct evidence of the assistant referring to an early thinking block dozens of tool-cycles later in another thinking block toward the end.

Gemini 2.5 Pro Specific Notes:
- Strict structured outputs
    - `config={"response_mime_type": "application/json", "response_schema": list[Recipe]}`
    - Refer to: https://ai.google.dev/gemini-api/docs/structured-output
    - Validation: Post‑validate all structured outputs with jsonschema or Pydantic; don’t assume perfection. (Google notes validators aren’t applied
- Function calling
    - Refer to: https://googleapis.github.io/python-genai/#function-calling
        - In case our framework impl makes it better for us to manually intercept function calls (e.g. to stay as event-based automaton system), you can do that and then it looks more like the the anthropic api with tool use.
    - Thinking with Interleaved Tool Use:
        - Refer to: https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#thinking
            - Preserving the thought signatures in the user request callbacks suggests that this works very similarly to Anthropic
        - The manual loop (what you’d write yourself if Automatic Function Calling disabled)
            1. Send generateContent with your tool declarations and prompt.
            2. If the response contains functionCall(s), execute those functions in your app.
            3. Build Part.from_function_response(name=..., response=...) for each call.
            4. Send another generateContent with:
                - the same configs and tools
                - the conversation messages thus far, and:
                - append the previous response model content (containing the last “thinking” thought signatures and the last function call(s)),
                - append your functionResponse parts (Google’s 2025 docs show these under role "user" in the contents, which the API accepts).
            5. Repeat until the model returns a normal text answer.
        - Interleaved reasoning with tools — Gemini 2.5 Pro behaves like Opus 4.1 (strong evidence for both models).
            - After a reasoning phase, the model can emit one or more functionCalls; once tool results are returned, it resumes with new reasoning before deciding whether to call more tools or produce text. Empirically you see thought-signature parts preceding the calls, then, after tool responses, new thought-signatures and another round of calls/text. This repeats across many cycles—matching Opus 4.1’s interleaved “think → tool → think → …” pattern.
        - Full reasoning continuity across tool cycles — same continuity model as Opus 4.1 (strong evidence for both models).
            - As long as you replay the entire prior model response (including thought-signatures) plus your function responses, the next turn continues a single, coherent chain of reasoning starting from the last user text prompt onward. Empirically, we find final outputs that are dependent on early internal thought state from many tool cycles ago, proving that the context window includes thought parts from many tool cycles ago - just like with Opus 4.1 when you fully replay history.
- Disable Thought Summaries always: they are not useful, and you don't want to accidentally allow them to be included as past thinking content.
- No particular tool specially used in training for bash or text editor like opus.
    - Re-use the `bash_tool` and `text_editor` but you need to provide tool explanation unlike opus.

Additional Notes:
- system prompts kept tiny and stable: agent’s role declaration, non‑negotiable rules/guardrails, output contract, meticulosity, verbosity/brevity, tool‑use policy (steer how often and when to use certain tools, beyond tool schema).
    - "you focus on performance optimization of the algorithm already select; do not propose new algorithms, just optimize impl using the one chosen."
    - "you must use tools to test performance and confirm speedups. You cannot just be speculative -- your results need to be backed up by numbers and you can admit lack of improvement."
- user prompt: (1) all the agent-specific context of the generic problem background (even if common to all instances this is still not system prompt), (2) the specific problem instance the agent is being invoked to do now.
- may help to be slightly repetitive of system prompt elements in user prompt to get better adherence.

Tools of note:
- human_in_loop()
    - becomes blocking for human input. Human can interject and this content will present forward guidance in the "tool" output.
    - various reasons why model may choose to invoke: (a) sign-off at key points; (b) lacking confidence and need guidance on the task.
- apply_diff_patch()
    - when a file needs to have a diff patch applied
    - file and diff patch provided as filepath
    - output path (usually different) where the new version goes
    - tries first to use git apply
    - if 1 or more hunk failures, reverts to text_editor if file is much larger than diff patch. If more than 30% of file affected, llm just rewrites the new file.
    - eases the burden on diff patch producers that don't actually need the patch to be applied

More investigation todo from:
- current frameworks
    - opencode
    - genagent
    - claude code reverse eng
- investigate other recent OSS frameworks (could be paradigms and methodologies, not necessarily code) for making agents, and libraries that my already be geared to doing what I have described here. What can be learned from those?


-----


# MVP Starting Software Architecture:

* `Function`: the central abstraction whereby code or agents are both abstracted as merely being kinds of function calls. Specification / metadata describing the agent or code.
    * Concrete subtypes must override abstract property `uses() -> List[Function]`, specifying any `Function`s that can be `invoke()`d by the `Function`.
    * `AgentFunction`: can be invoked by any `Function`.
        * user subtypes to define their own agents (could use abstract properties that user must override).
        * subtype must specify: input vars, system prompt, templated user prompt (var substitution). Each var may be given as strings or filepaths (upon instantiation of the agent, files would have to be loaded and then substitution done by the runner infra instead of asking the agent to do it).
        * subtype specifies `uses: List[Function]` — the Functions it can invoke via tool calls
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
* `Runtime`: top-level runner and state management
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
        * `list_toplevel() -> List[Node]`
        * `get_subtree(node_id: int)`
        * To prevent race conditions, consumers should use `Runtime` to query state and do top-level invocations.
* `RunContext`: common framework interface used by both framework consumers and framework internal impl to invoke `Function`s.
    * serves at least as the interface for:
        1. top-level task invocation; called by an app that is consuming the framework and a collection of `Function`s (the app or someone else may define these); access via `Runtime.get_ctx()`.
        2. python code for user-defined or framework-builtin `CodeFunction`s that invoke other `Function`s.
        3. when some framework component needs to handle agents doing tool calls, that component delegates invocation to the `RunContext`.
        * e.g. they all use: `ctx.invoke(fn: Function, args)`
    * every `Function` invocation has a `RunContext` given to it, providing the interface, but also tracking the particular `Function` using it.
        * when a `Function` invokes another `Function` (including when framework handles `AgentFunction` invoking any `Function` via tool call), the `RunContext` knows its associated invoking `Node` (identity of the caller) and causes creation of the invoked `Node`.
            * this information is used to construct the directed edges relationships of the `Node` tree. A single top-level task invocation is the parent `Node` of a tree.
    * Each top-level `ctx.invoke()` (by consuming app) initiates one tree of `Node`s where the parent `Node` of the tree is the top-level Task.
        * Each top-level Task is an independent tree with `Node`s disjoint from those originating from other top-level tasks.
        * Each top-level Task may originate from the invocation of **any** `Function` that was registered with `start_runtime()`, thus being coarse-grained tasks or fine-grained tasks at the top level.
            * General idea: Fine-grained top-level Tasks would appear as shallow trees, that may be comparable to the deepest subtrees of a coarse-grained top-level Task that decomposes into the former -- the latter being a broader-scope task that needs to solve the former's scope of problem perhaps as a mere sub-sub-Task.
    * Fields:
        * `node: Node`: a reference to the particular `Node` identifying this specific `Function` invocation.
        * `runtime: Runtime`: a reference to the shared `Runtime`.
    * Narrow Scope: `RunContext` is just a mechanism to pass on `Function` invocation directives to the `Runtime` to act on them.
* `Node`: abstract object that represents the invocation of a `Function` (which we also call a "Task").
    * `AgentNode`: represents and manages the state and running of an `AgentFunction` invocation.
        * `AnthropicAgentNode`
            * particular implementation when the `AgentFunction` is invoked with Anthropic LLM (e.g. Opus 4.1).
        * `GeminiAgentNode`
            * particular implementation when the `AgentFunction` is invoked with Gemini LLM (e.g. Gemini Pro 2.5).
        * Tracks history of LLM session thus far (which it also uses in tool cycle when doing follow-up request)
            * Subtypes `AnthropicAgentNode` and `GeminiAgentNode` store and use the SDK-specific types in their internal impl.
        * `node.get_transcript() -> List[TranscriptPart]`
            * Subtypes must implement; they must convert the SDK-specific types in the transcription they are tracking to the framework-common `TranscriptPart`s. They never convert types in the reverse direction.
        * `node.get_children() -> List[Node | List[Node]]` (single | parallel tool call).
            * Always ordered to reflect the sequence in which `Function`s were invoked. Parallel tool calls by LLM (parallel `Function` invocations) always shown tied by being in same inner List.
            * Outer list index gives the invocation sequence.
        * Has states (Waiting, Running, Success, Error) but also sub-state including tool use (`Function` invocation) that it is waiting on.
        * `AgentNode` is completed once it returns final assistant text or has its `return_result` or `raise_exception` called (if it has those tools).
    * `CodeNode`: represents and manages the state and running of a `CodeFunction` invocation.
        * Considerably simpler than `AgentFunction` because there are few states (Waiting, Running, Success, Error) and simple function call (unlike LLM session complexity).
        * Only implements `node.get_children() -> List[Node]`.
        * `CodeNode` is completed once it either returns or raises.
    * Fields / Properties:
        * `id: int`: monotonically increasing unique identifier when Node is to be used as key in any lookup. This is one and the same as "task id".
        * `function: Function`: which `Function` the `Node` is an instance of.
        * `inputs: Dict`: (dynamic). What the inputs were for the invocation.
        * `outputs: Optional[Dict]`: (dynamic). What the outputs were from the run (if finished).
        * `exception: Optional[Exception]`: the exception, if there was an exception.
        * `state: NodeState`: (Waiting, Running, Success, Error) enum
        * `waiting_on: Optional[Node]`: the `Node` of the `Function` being waited on if we are currently in the process of calling one.
* `TranscriptPart`
    * parts that are common to the model-specific SDKs in concept.
    * `UserText`
    * `ModelText`
    * `ToolUse`
    * `ToolResult`
    * `ThinkingBlock`
        * including both redacted and non-redacted
        * includes `ThinkingSignature`
    * On every follow-up call, replay the full history in original order.
* `Ensemble`
    * Is a `CodeFunction` that decorates any `AgentFunction` to do parallel independent invocations followed by reconciliation.
        * First phase: each `AgentFunction` call proceeds as normal, with args forwarded for normal user prompt substitution.
        * Second phase: same system prompt and substituted user prompt; append to the user prompt each of the completions along with a reconciliation instruction.
    * Given any `AgentFunction`, mostly users will construct one from built-in factory facility (ctor directly):
        * `Ensemble(agent: AgentFunction, instances: Dict[Provider, int], name: Optional[str] = None, reconcile_by: Optional[Provider] = None)`
            * `instances`: how many parallel invocations of `AgentFunction` to do with each model.
        * User uses this when defining their `Function`s.
    * Automatically has a valid inner Callable like any `CodeFunction` that does the ensembling phases.
* `SessionBag`
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
* Exception Model
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
    * Framework support mechanisms:
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
* **Deferred features** (bucket list of nice-to-haves):
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
        * Enforce at `start_runtime()`.
        * Ensure each `Function` has legal references to other `Function`s.
        * Reject if not a DAG.
        * Reject if function type annotations do not match the spec in `CodeFunction`.