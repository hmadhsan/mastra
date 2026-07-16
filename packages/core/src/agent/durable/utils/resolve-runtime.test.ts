import { afterEach, describe, expect, it, vi } from 'vitest';
import { ProcessorRunner } from '../../../processors/runner';
import { MessageList } from '../../message-list';
import { globalRunRegistry } from '../run-registry';
import { resolveRuntimeDependencies } from './resolve-runtime';

const RUN_ID = 'cross-process-agent-context';

afterEach(() => {
  if (globalRunRegistry.has(RUN_ID)) globalRunRegistry.delete(RUN_ID);
});

describe('resolveRuntimeDependencies', () => {
  it('rehydrates the agent so processors can access its memory cross-process', async () => {
    const memory = {};
    const agent = {
      getToolsForExecution: vi.fn().mockResolvedValue({}),
      getModel: vi.fn().mockResolvedValue({ specificationVersion: 'v2', provider: 'test', modelId: 'test' }),
      getModelList: vi.fn().mockResolvedValue(undefined),
      getMemory: vi.fn().mockResolvedValue(memory),
      getWorkspace: vi.fn().mockResolvedValue(undefined),
      listInputProcessors: vi.fn().mockResolvedValue([]),
      __listLLMRequestProcessors: vi.fn().mockResolvedValue([]),
      listOutputProcessors: vi.fn().mockResolvedValue([]),
      listErrorProcessors: vi.fn().mockResolvedValue([]),
    };
    const mastra = {
      getAgentById: vi.fn().mockReturnValue(agent),
      getLogger: vi.fn().mockReturnValue(undefined),
    };

    globalRunRegistry.set(RUN_ID, {
      isPlaceholder: true,
      tools: {},
      model: undefined,
    } as any);

    const resolved = await resolveRuntimeDependencies({
      mastra: mastra as any,
      runId: RUN_ID,
      agentId: 'memory-agent',
      input: {
        runId: RUN_ID,
        agentId: 'memory-agent',
        messageListState: new MessageList().serialize(),
        state: {
          threadId: 'thread-1',
          resourceId: 'resource-1',
          threadExists: false,
        },
      } as any,
    });

    expect(resolved.agent).toBe(agent);
    expect(globalRunRegistry.get(RUN_ID)?.agent).toBe(agent);

    let processorMemory: unknown;
    const runner = new ProcessorRunner({
      inputProcessors: [
        {
          id: 'cross-process-memory-reader',
          processInputStep: async ({ agent: processorAgent }) => {
            processorMemory = await processorAgent?.getMemory();
          },
        },
      ],
      outputProcessors: [],
      logger: { debug: vi.fn(), warn: vi.fn(), error: vi.fn() } as any,
      agentName: 'memory-agent',
      agent: resolved.agent,
    });
    await runner.runProcessInputStep({
      messageList: resolved.messageList,
      stepNumber: 0,
      steps: [],
      model: resolved.model,
    });

    expect(processorMemory).toBe(memory);
  });
});
