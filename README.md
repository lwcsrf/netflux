`netflux` is :

- minimalist python framework for authoring custom agentic applications. The core idea is to treat agents like any code function in imperative programming: you have inputs, outputs, logic composition (by calling on building blocks of other functions), and side effects on structures.
- our goal was to build a framework that is semantically flexible to do the equivalent of both workflows and dynamic, open-ended problem solvers. Fundamentally, we believe hierarchical Task Decomposition is important to doing this effectively, the same cohesion and reusability of libraries and utility functions in traditional programming.
- explain the central abstraction `Function`.
- explain `CodeFunction`
- explain `AgentFunction`. and why the abstraction of treating it like a Function works well. We may often say "Agent" and we mean an instance of `AgentFunction`.
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

Tips & Tricks

- The framework tries to make it easy to do effective context engineering. Usually higher-level agents will have the role of orchestration or workflow. Lower-level agents will solve more concrete patterns of problem. As LLMs become more sophisticated, a single agent can take on a broader set of responsibilities (more Functions as its disposal, and longer-running). Partition sub-tasks as fine-grained as needed but don't over-partition unnecessarily as this can degrade your evals.
- agents may use files for input(s)/output(s). Input filepaths would be given as args, and output filepaths may be returned (Write File tool used prior to returning).
- structured outputs are discouraged since LLMs are sophisticated enough to parse unstructured outputs from their sub-tasks. However, sometimes strict structured outputs are critical, and this can be enforced by defining `CodeFunction`s where the arguments are the schema and the Callable performs serialization and/or verification, depending on the reason for the structured output, and returns empty or provides a filepath with the serialized data, etc. You can leverage the framework's Exceptions Model to propagate an Exception if verification fails.
- When authoring agents, place the common prompt before the specifics of the task instance. This is because LLMs are known to pay greater attention to the beginning and end of the context window. Particularly, when giving background information, whatever most heavily will influence the specific actions the agent will take should be placed closer to the end of the prompt.
- human_in_loop()
    - becomes blocking for human input. Human can interject and this content will present forward guidance in the "function" output.
    - implement the hook to human UI using whatever mechanism particular to your application.
    - various reasons why model may choose to invoke: (a) sign-off at key points; (b) lacking confidence and need guidance on the task; (c) on the verge of RaiseException and seeking opinion of what to try before doing so.

- Agent instance token accounting. Discuss in a couple sentence what we track. Refer to relevant section below.

- Framework shall use a process-wide global semaphore to limit the number of concurrent requests to any model api in flight. When an agent or tool is invoked, it is given a shared object containing a reference to the semaphore and a boolean `has_sem`, which is inherited from the top-level task. Some non-agentic tools that know they are long-running can courteously give up their lease on the semaphore. For example, the human_in_loop tool should do this. After unwinding, when an agent once again needs to make a request to a model, it would need to re-acquire the semaphore lease if it was given up. By default, we do not give up the semaphore in order to make more effective use of prompt caching (which has ephemeral ttl) and to encourage front-of-line agent throughput when many top-level tasks have been enqueued.



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



-----


# Entities & Architecture

## `Function`

`Function` is the central abstraction whereby code or agents are both abstracted as merely being kinds of function calls. Specification / metadata describing the agent or code.

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
    * `list_toplevel() -> List[Node]`
    * `get_subtree(node_id: int)`
    * To prevent race conditions, consumers should use `Runtime` to query state and do top-level invocations.

## `RunContext`

`RunContext` is a common framework interface used by both framework consumers and framework internal impl to invoke `Function`s.

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

## `TranscriptPart`

`TranscriptPart` represents each of the parts that are common to the model-specific SDKs in concept. The subtypes:

* `UserText`
* `ModelText`
* `ToolUse`
* `ToolResult`
* `ThinkingBlock`
    * including both redacted and non-redacted
    * includes `ThinkingSignature`
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
    * Token accounting: maintain a cumulative `TokenUsage` throughout the agent loop and update it on every request/response iteration.
        * Refer to the `TokenUsage` class fields in code for exact semantics. Keep all applicable fields up to date each turn, mapping from the provider SDK usage metadata.
        * Inputs: increment `input_tokens_cache_read`, `input_tokens_cache_write` (if applicable), `input_tokens_regular`, and `input_tokens_total`.
        * Outputs: increment `output_tokens_reasoning` and/or `output_tokens_text` when the provider exposes a breakdown; always increment `output_tokens_total`.
        * Each provider subtype must expose a `token_usage` property returning the cumulative `TokenUsage`. See `AnthropicAgentNode` and `GeminiAgentNode` for reference.
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
- Consumer Watcher loop: call `node.watch(as_of_seq=prev_seq+1)` to block until there is a newer snapshot (where `touch_seqno >= as_of_seq`) and then receive the latest `NodeView` for that subtree. Start with `prev_seq = 0` and after each update set `prev_seq = view.update_seqnum` to continue receiving atomic updates. This is similar to etcd/zookeeper watchers, except the `Runtime` keeps only the latest view available. Watcher loops should be used for event-driven UI.
- Top-level views: use `runtime.list_toplevel_views()` to get a consistent snapshot of all root tasks at once; `runtime.get_view(node_id)` returns the latest snapshot for any node without blocking.
- Why: this isolates observers from partial, in-flight mutations during execution and guarantees each delivered view is a self-consistent global snapshot of the tree at a specific sequence number.

## Deferred Features

This is a bucket list of nice-to-haves.

- `ApplyDiffPatch(CodeFunction)`
    - when a file needs to have a diff patch applied (the thing that a git diff patch looks like).
    - file and diff patch provided as filepath or as the content itself.
    - output path (usually different) where the new version goes
    - tries first to use git apply on each hunk using bash.
    - for any hunk that fails, reverts to text_editor for those hunks.
    - if one of those hunks still fails, can try searching (e.g. using grep or other) to get context and see what might be going on.
    - raises exception if one or more hunks is significantly different as the Before or even fuzzy matching is not possible. Only be lenient to very minor things like extra newline or minor typo but otherwise unambiguous.
    - eases the burden on diff patch producers that don't actually need the patch to be applied immediately.

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
