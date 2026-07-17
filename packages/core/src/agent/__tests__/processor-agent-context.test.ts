import type { LanguageModelV2 } from '@ai-sdk/provider-v5';
import { APICallError } from '@internal/ai-sdk-v5';
import { MockLanguageModelV2, convertArrayToReadableStream } from '@internal/ai-sdk-v5/test';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { EventEmitterPubSub } from '../../events/event-emitter';
import { Mastra } from '../../mastra';
import { MockMemory } from '../../memory/mock';
import type { ErrorProcessor, InputProcessor, OutputProcessor } from '../../processors';
import { ProcessorStepInputSchema, ProcessorStepOutputSchema } from '../../processors/step-schema';
import { RequestContext } from '../../request-context';
import { createStep, createWorkflow } from '../../workflows';
import { PUBSUB_SYMBOL } from '../../workflows/constants';
import { Agent } from '../agent';
import { createDurableAgent } from '../durable/create-durable-agent';
import { globalRunRegistry } from '../durable/run-registry';
import { createDurableLLMExecutionStep } from '../durable/workflows/steps/llm-execution';

function createTextModel(): LanguageModelV2 {
  return new MockLanguageModelV2({
    doGenerate: async () => ({
      rawCall: { rawPrompt: null, rawSettings: {} },
      finishReason: 'stop',
      usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
      content: [{ type: 'text', text: 'Hello' }],
      warnings: [],
    }),
    doStream: async () => ({
      stream: convertArrayToReadableStream([
        { type: 'stream-start', warnings: [] },
        { type: 'text-start', id: 'text-1' },
        { type: 'text-delta', id: 'text-1', delta: 'Hello' },
        { type: 'text-end', id: 'text-1' },
        {
          type: 'finish',
          finishReason: 'stop',
          usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
        },
      ]),
      rawCall: { rawPrompt: null, rawSettings: {} },
    }),
  });
}

function createRecoveringTextModel(): LanguageModelV2 {
  let callCount = 0;
  return new MockLanguageModelV2({
    doStream: async () => {
      callCount++;
      if (callCount === 1) {
        throw new APICallError({
          message: 'Invalid request',
          url: 'https://example.com/model',
          requestBodyValues: {},
          statusCode: 400,
          isRetryable: false,
        });
      }
      return {
        stream: convertArrayToReadableStream([
          { type: 'stream-start', warnings: [] },
          { type: 'text-start', id: 'text-1' },
          { type: 'text-delta', id: 'text-1', delta: 'Recovered' },
          { type: 'text-end', id: 'text-1' },
          {
            type: 'finish',
            finishReason: 'stop',
            usage: { inputTokens: 1, outputTokens: 1, totalTokens: 2 },
          },
        ]),
        rawCall: { rawPrompt: null, rawSettings: {} },
      };
    },
  });
}

type AgentObservation = { hook: string; agent: unknown };

function createAgentAwareInputProcessor(observations: AgentObservation[]): InputProcessor {
  return {
    id: 'agent-aware-input',
    processInput: async ({ agent, messages }) => {
      observations.push({ hook: 'processInput', agent });
      return messages;
    },
    processInputStep: async ({ agent }) => {
      observations.push({ hook: 'processInputStep', agent });
    },
    processLLMRequest: async ({ agent, prompt }) => {
      observations.push({ hook: 'processLLMRequest', agent });
      return { prompt };
    },
    processLLMResponse: async ({ agent }) => {
      observations.push({ hook: 'processLLMResponse', agent });
    },
  };
}

function createAgentAwareOutputProcessor(observations: AgentObservation[]): OutputProcessor {
  return {
    id: 'agent-aware-output',
    processOutputStream: async ({ agent, part }) => {
      observations.push({ hook: 'processOutputStream', agent });
      return part;
    },
    processOutputResult: async ({ agent, messages }) => {
      observations.push({ hook: 'processOutputResult', agent });
      return messages;
    },
    processOutputStep: async ({ agent, messages }) => {
      observations.push({ hook: 'processOutputStep', agent });
      return messages;
    },
  };
}

function createAgentAwareErrorProcessor(observations: AgentObservation[]): ErrorProcessor {
  return {
    id: 'agent-aware-error',
    processAPIError: async ({ agent }) => {
      observations.push({ hook: 'processAPIError', agent });
      return { retry: true };
    },
  };
}

function expectAgentForHooks(observations: AgentObservation[], agent: Agent, expectedHooks: string[]) {
  expect(new Set(observations.map(({ hook }) => hook))).toEqual(new Set(expectedHooks));
  expect(observations.every(observation => observation.agent === agent)).toBe(true);
}

describe('processor agent context', () => {
  const pubsubs: EventEmitterPubSub[] = [];
  const runIds: string[] = [];

  afterEach(async () => {
    await Promise.all(pubsubs.splice(0).map(pubsub => pubsub.close()));
    for (const runId of runIds.splice(0)) {
      if (globalRunRegistry.has(runId)) globalRunRegistry.delete(runId);
    }
  });

  it.each(['generate', 'stream'] as const)(
    'provides the owning agent to every input and output processor hook during %s',
    async method => {
      const inputObservations: AgentObservation[] = [];
      const outputObservations: AgentObservation[] = [];
      const agent = new Agent({
        id: `processor-context-agent-${method}`,
        name: 'Processor context agent',
        instructions: 'Respond briefly.',
        model: createTextModel(),
        inputProcessors: [createAgentAwareInputProcessor(inputObservations)],
        outputProcessors: [createAgentAwareOutputProcessor(outputObservations)],
      });

      if (method === 'generate') {
        await agent.generate('Hello');
      } else {
        const result = await agent.stream('Hello');
        await result.consumeStream();
      }

      expectAgentForHooks(inputObservations, agent, [
        'processInput',
        'processInputStep',
        'processLLMRequest',
        'processLLMResponse',
      ]);
      expectAgentForHooks(outputObservations, agent, [
        'processOutputStream',
        'processOutputResult',
        'processOutputStep',
      ]);
    },
  );

  it('provides the owning agent to processors inside a prebuilt nested workflow', async () => {
    let processorAgent: unknown;
    const workflowProcessor: InputProcessor = {
      id: 'nested-workflow-agent-aware-input',
      processInput: async ({ agent, messages }) => {
        processorAgent = agent;
        return messages;
      },
    };
    const processorWorkflow = createWorkflow({
      id: 'agent-context-prebuilt-processor-workflow',
      inputSchema: ProcessorStepInputSchema,
      outputSchema: ProcessorStepOutputSchema,
    })
      .then(createStep(workflowProcessor))
      .commit();
    const agent = new Agent({
      id: 'nested-workflow-processor-context-agent',
      name: 'Nested workflow processor context agent',
      instructions: 'Respond briefly.',
      model: createTextModel(),
      inputProcessors: [
        {
          id: 'direct-agent-aware-input',
          processInput: async ({ messages }) => messages,
        },
        processorWorkflow,
      ],
    });

    await agent.generate('Hello');

    expect(processorAgent).toBe(agent);
  });

  it('keeps agents isolated when they share one prebuilt processor workflow', async () => {
    const observedAgents = new Map<string, unknown>();
    let resolveFirstProcessorEntered!: () => void;
    const firstProcessorEntered = new Promise<void>(resolve => {
      resolveFirstProcessorEntered = resolve;
    });
    let resolveRelease!: () => void;
    const release = new Promise<void>(resolve => {
      resolveRelease = resolve;
    });

    const sharedProcessor: InputProcessor = {
      id: 'shared-workflow-agent-aware-input',
      processInput: async ({ agent, messages }) => {
        const owner = (agent as Agent | undefined)?.id ?? 'unknown';
        resolveFirstProcessorEntered();
        // Hold the first run open until the second agent has resolved its
        // processors, to prove late binding cannot clobber an in-flight run.
        await release;
        observedAgents.set(owner, agent);
        return messages;
      },
    };
    const sharedWorkflow = createWorkflow({
      id: 'shared-processor-workflow',
      inputSchema: ProcessorStepInputSchema,
      outputSchema: ProcessorStepOutputSchema,
    })
      .then(createStep(sharedProcessor))
      .commit();

    const agentA = new Agent({
      id: 'shared-workflow-agent-a',
      name: 'Shared workflow agent A',
      instructions: 'Respond briefly.',
      model: createTextModel(),
      inputProcessors: [sharedWorkflow],
    });
    const agentB = new Agent({
      id: 'shared-workflow-agent-b',
      name: 'Shared workflow agent B',
      instructions: 'Respond briefly.',
      model: createTextModel(),
      inputProcessors: [sharedWorkflow],
    });

    const generateA = agentA.generate('Hello from A');
    await firstProcessorEntered;
    // Resolving agent B's processors while agent A's run is mid-flight must
    // not rebind the shared workflow away from agent A.
    const generateB = agentB.generate('Hello from B');
    resolveRelease();
    await Promise.all([generateA, generateB]);

    expect(observedAgents.get('shared-workflow-agent-a')).toBe(agentA);
    expect(observedAgents.get('shared-workflow-agent-b')).toBe(agentB);
  });

  it.each(['generate', 'stream'] as const)(
    'provides the wrapped agent to every same-process durable processor hook during %s',
    async method => {
      const inputObservations: AgentObservation[] = [];
      const outputObservations: AgentObservation[] = [];
      const baseAgent = new Agent({
        id: `durable-processor-context-agent-${method}`,
        name: 'Durable processor context agent',
        instructions: 'Respond briefly.',
        model: createTextModel(),
        inputProcessors: [createAgentAwareInputProcessor(inputObservations)],
        outputProcessors: [createAgentAwareOutputProcessor(outputObservations)],
      });
      const pubsub = new EventEmitterPubSub();
      pubsubs.push(pubsub);
      const durableAgent = createDurableAgent({ agent: baseAgent, pubsub });

      if (method === 'generate') {
        await durableAgent.generate('Hello');
      } else {
        const { output, cleanup } = await durableAgent.stream('Hello');
        await output.consumeStream();
        cleanup();
      }

      expectAgentForHooks(inputObservations, baseAgent, [
        'processInput',
        'processInputStep',
        'processLLMRequest',
        'processLLMResponse',
      ]);
      expectAgentForHooks(outputObservations, baseAgent, [
        'processOutputStream',
        'processOutputResult',
        'processOutputStep',
      ]);
    },
  );

  it('rehydrates the owning agent for cross-process durable processor memory access', async () => {
    const memory = new MockMemory();
    let processorAgent: unknown;
    let processorMemory: unknown;
    const baseAgent = new Agent({
      id: 'cross-process-processor-context-agent',
      name: 'Cross-process processor context agent',
      instructions: 'Respond briefly.',
      model: createTextModel(),
      inputProcessors: [
        {
          id: 'cross-process-memory-reader',
          processInputStep: async ({ agent }) => {
            processorAgent = agent;
            processorMemory = await agent?.getMemory();
          },
        },
      ],
    });
    const mastra = new Mastra({ agents: { baseAgent } });
    const pubsub = new EventEmitterPubSub();
    pubsubs.push(pubsub);
    const durableAgent = createDurableAgent({ agent: baseAgent, pubsub });
    const preparation = await durableAgent.prepare('Hello');
    runIds.push(preparation.runId);

    vi.spyOn(baseAgent, 'getMemory').mockResolvedValue(memory as any);
    globalRunRegistry.set(preparation.runId, {
      isPlaceholder: true,
      tools: {},
      model: undefined,
    } as any);

    const step = createDurableLLMExecutionStep();
    await (step as any).execute({
      inputData: preparation.workflowInput,
      mastra,
      requestContext: new RequestContext(),
      tracingContext: {},
      abortSignal: undefined,
      [PUBSUB_SYMBOL]: pubsub,
    });

    expect(processorAgent).toBe(baseAgent);
    expect(processorMemory).toBe(memory);
  });

  it('provides the owning agent to standard API error processors', async () => {
    const observations: AgentObservation[] = [];
    const agent = new Agent({
      id: 'error-processor-context-agent',
      name: 'Error processor context agent',
      instructions: 'Respond briefly.',
      model: createRecoveringTextModel(),
      errorProcessors: [createAgentAwareErrorProcessor(observations)],
    });

    const result = await agent.stream('Hello');
    await result.consumeStream();

    expectAgentForHooks(observations, agent, ['processAPIError']);
  });

  it('provides the wrapped agent to durable API error processors', async () => {
    const observations: AgentObservation[] = [];
    const baseAgent = new Agent({
      id: 'durable-error-processor-context-agent',
      name: 'Durable error processor context agent',
      instructions: 'Respond briefly.',
      model: createRecoveringTextModel(),
      errorProcessors: [createAgentAwareErrorProcessor(observations)],
    });
    const pubsub = new EventEmitterPubSub();
    pubsubs.push(pubsub);
    const durableAgent = createDurableAgent({ agent: baseAgent, pubsub });

    await durableAgent.generate('Hello');

    expectAgentForHooks(observations, baseAgent, ['processAPIError']);
  });
});
