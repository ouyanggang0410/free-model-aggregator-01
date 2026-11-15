// main.ts — the free-model aggregator server.
//
// Flow: read the brief, probe every provider's catalog route, build one
// catalog, listen on a free port, register with the env, answer the client's
// requests, and poll the env for the end of the trial. `node main.ts --list`
// prints the same catalog the server registers, byte for byte, and exits 0.
//
// Routing: for each request the server ranks every free model that meets the
// request's requirements by the quality table's rating, tries the best first,
// and on a provider failure (402/429/5xx) moves to the next model. A provider
// that answers "budget is spent" or "is out" is never called again — those
// are permanent; a provider that "refused this call" failed only that one
// call and stays in the ranking. Calls are spread across equal models by
// how few calls a provider has served so far.
//
// Determinism: no clock is read, no randomness is used. Every choice comes
// from the brief, the request, and the providers' answers, so a replay of
// the same trial makes the same choices.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  probeOne,
  parseBrief,
  ratingFor,
  isFreeModel,
  capsFor,
  meetsNeed,
  joinBase,
  authHeaders,
  type Brief,
  type Catalog,
  type CatalogModel,
  type DirProvider,
  type Need,
  type ProbedProvider,
  type Json,
} from './lib/brief.ts';
import {
  upstreamBody,
  parseProviderAnswer,
  renderOpenAIJson,
  renderOpenAIStream,
  renderAnthropicJson,
  renderAnthropicStream,
  modelsListJson,
  errorJson,
  clientJsonMode,
  clientMessages,
  type ClientApi,
  type ProviderWire,
  type ProviderAnswer,
} from './lib/translate.ts';

const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROVIDER_TIMEOUT_MS = 10000;
const POLL_INTERVAL_MS = 100;

function log(...parts: unknown[]): void {
  console.error(`[aggregator] ${parts.map((p) => String(p)).join(' ')}`);
}

interface Loaded {
  brief: Brief;
  probes: ProbedProvider[];
  catalog: Catalog;
}

/** Read the brief and probe every provider it lists. */
async function load(): Promise<Loaded> {
  const brief = parseBrief(PROMPT_FILE);
  if (brief.problem !== null) throw new Error(brief.problem);
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const probes = await Promise.all(brief.directory.providers.map((p) => probeOne(ENV_URL, p)));
  for (const probe of probes) {
    if (probe.ok) log(`catalog ${probe.provider.id}: ${probe.models.length} models`);
    else log(`catalog ${probe.provider.id}: FAILED (${probe.problem})`);
  }
  return { brief, probes, catalog: buildCatalog(probes, brief.ratings) };
}

interface Candidate {
  provider: DirProvider;
  wire: ProviderWire;
  model: string;
  rating: number;
  calls: number;
}

interface CallOutcome {
  status: number;
  text: string;
  networkError: boolean;
}

/** The state of one running server: what it has learned about the providers. */
class Aggregator {
  private brief: Brief;
  private probes: ProbedProvider[];
  private catalog: Catalog;
  /** Calls made to each provider so far, for spreading load across equals. */
  private calls = new Map<string, number>();
  /** Providers a permanent refusal has taken out of the ranking. */
  private exhausted = new Set<string>();
  /** Chat requests served so far, so the workload plan can be matched by arrival order. */
  private arrivals = 0;
  /** One request at a time: a provider serves one call at a time, and this server never makes two. */
  private queue: Promise<void> = Promise.resolve();

  constructor(loaded: Loaded) {
    this.brief = loaded.brief;
    this.probes = loaded.probes;
    this.catalog = loaded.catalog;
  }

  get catalogDoc(): Catalog {
    return this.catalog;
  }

  /** The free model ids this server registered, deduplicated and sorted. */
  private freeModelIds(): string[] {
    const ids = new Set<string>();
    for (const p of this.catalog.providers) for (const m of p.models) if (m.free) ids.add(m.id);
    return [...ids].sort();
  }

  /** The requirements of the request arriving now, from the plan or the body itself. */
  private needFor(api: ClientApi, body: Json, arrival: number): Need {
    const plan = this.brief.plan;
    if (plan !== null && arrival < plan.items.length) {
      const item = plan.items[arrival];
      if (item.api === api) return item.requires;
    }
    const need: Need = {};
    const messages = clientMessages(api, body);
    const hasImage = messages.some((m) => m.parts.some((p) => p.type === 'image'));
    if (hasImage) need.vision = true;
    if (Array.isArray(body.tools) && body.tools.length > 0) need.tools = true;
    if (clientJsonMode(api, body)) need.json = true;
    return need;
  }

  /** Every free model that can serve this request, best rating first. */
  private candidatesFor(api: ClientApi, body: Json, arrival: number): Candidate[] {
    const need = this.needFor(api, body, arrival);
    const jsonMode = clientJsonMode(api, body);
    const out: Candidate[] = [];
    for (const probe of this.probes) {
      if (!probe.ok) continue;
      if (this.exhausted.has(probe.provider.id)) continue;
      for (const model of probe.models) {
        const row = ratingFor(this.brief.ratings, probe.provider.id, model.id);
        if (!isFreeModel(model, row)) continue;
        if (!meetsNeed(need, capsFor(model, row))) continue;
        if (jsonMode && probe.provider.wire === 'anthropic') continue;
        out.push({ provider: probe.provider, wire: probe.provider.wire, model: model.id, rating: row?.ratingMilli ?? 0, calls: this.calls.get(probe.provider.id) ?? 0 });
      }
    }
    out.sort((a, b) => b.rating - a.rating || a.calls - b.calls || (a.provider.id < b.provider.id ? -1 : 1) || (a.model < b.model ? -1 : 1));
    return out;
  }

  /** One call to one provider, in its own wire. A 400/401 is a bug, not a retryable failure. */
  private async callProvider(candidate: Candidate, upstream: Json, stream: boolean): Promise<CallOutcome> {
    const { provider, wire, model } = candidate;
    const base = joinBase(ENV_URL, provider.base);
    let url: string;
    if (wire === 'openai') url = `${base}/v1/chat/completions`;
    else if (wire === 'anthropic') url = `${base}/v1/messages`;
    else url = `${base}/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    const headers: Record<string, string> = {
      'content-type': 'application/json',
      accept: stream ? 'text/event-stream' : 'application/json',
      ...authHeaders(provider.auth, provider.key),
    };
    if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
    try {
      const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(upstream), signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS) });
      return { status: res.status, text: await res.text(), networkError: false };
    } catch (err) {
      return { status: 0, text: `call to ${provider.id} failed: ${(err as Error).message}`, networkError: true };
    }
  }

  /** The failure message a provider's error body carries, when it has one. */
  private failureMessage(outcome: CallOutcome): string {
    if (outcome.networkError) return outcome.text;
    try {
      const parsed: unknown = JSON.parse(outcome.text);
      if (parsed !== null && typeof parsed === 'object' && 'error' in parsed) {
        const err = (parsed as { error?: unknown }).error;
        if (err !== null && typeof err === 'object' && 'message' in err) return String((err as { message?: unknown }).message ?? '');
        if (typeof err === 'string') return err;
      }
    } catch {
      // Not a JSON error body; fall through to the raw text.
    }
    return outcome.text;
  }

  /**
   * Serve one chat request: try the candidates best-first, failing over on a
   * provider refusal, and render the first answer that comes back in the
   * client's wire, JSON or streamed as the client asked.
   */
  private async serveChat(api: ClientApi, body: Json): Promise<{ status: number; payload: Json | string; sse: boolean }> {
    const stream = body.stream === true;
    const myArrival = this.arrivals;
    this.arrivals += 1;
    const candidates = this.candidatesFor(api, body, myArrival);
    if (candidates.length === 0) {
      const { status, body: errBody } = errorJson(api, 503, 'no free model meets this request', 'no_model_available');
      return { status, payload: errBody, sse: false };
    }
    log(`request ${myArrival + 1} (${api}${stream ? ', stream' : ''}): ${candidates.length} candidates, best ${candidates[0].provider.id}/${candidates[0].model} rating ${candidates[0].rating}`);
    const tried = new Set<string>();
    for (const candidate of candidates) {
      if (tried.has(candidate.provider.id)) continue;
      tried.add(candidate.provider.id);
      const upstream = upstreamBody(api, candidate.wire, body, candidate.model, stream);
      const outcome = await this.callProvider(candidate, upstream, stream);
      this.calls.set(candidate.provider.id, (this.calls.get(candidate.provider.id) ?? 0) + 1);
      if (outcome.status >= 200 && outcome.status < 300) {
        const answer: ProviderAnswer = parseProviderAnswer(candidate.wire, outcome.text);
        log(`request ${myArrival + 1}: served by ${candidate.provider.id}/${candidate.model} (usage ${answer.usage.input}+${answer.usage.output})`);
        if (api === 'openai') {
          return stream
            ? { status: 200, payload: renderOpenAIStream(candidate.model, answer), sse: true }
            : { status: 200, payload: renderOpenAIJson(candidate.model, answer), sse: false };
        }
        return stream
          ? { status: 200, payload: renderAnthropicStream(candidate.model, answer), sse: true }
          : { status: 200, payload: renderAnthropicJson(candidate.model, answer), sse: false };
      }
      const message = this.failureMessage(outcome);
      log(`request ${myArrival + 1}: ${candidate.provider.id} refused (HTTP ${outcome.status}): ${message}`);
      if (message.includes('budget is spent') || message.includes('is out')) {
        this.exhausted.add(candidate.provider.id);
        log(`${candidate.provider.id} is exhausted; it will not be called again`);
      }
      if (outcome.status === 400 || outcome.status === 401) {
        log(`request ${myArrival + 1}: ${candidate.provider.id} answered ${outcome.status} — a request the provider rejects; moving on`);
      }
    }
    const { status, body: errBody } = errorJson(api, 503, 'all providers refused this request', 'all_providers_failed');
    return { status, payload: errBody, sse: false };
  }

  /** Read a request body as JSON, or null when it is not one. */
  private async readJsonBody(req: IncomingMessage): Promise<Json | null> {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    try {
      const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      return parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed) ? (parsed as Json) : null;
    } catch {
      return null;
    }
  }

  private send(res: ServerResponse, status: number, payload: Json | string, sse: boolean): void {
    res.writeHead(status, { 'content-type': sse ? 'text/event-stream' : 'application/json' });
    res.end(typeof payload === 'string' ? payload : JSON.stringify(payload));
  }

  /** One HTTP request, queued behind any request already being served. */
  handle(req: IncomingMessage, res: ServerResponse): void {
    const run = this.queue.then(async () => {
      const url = new URL(req.url ?? '/', 'http://localhost');
      const path = url.pathname;
      if (req.method === 'GET' && path === '/v1/models') {
        const wire: ClientApi = req.headers['x-api-key'] !== undefined ? 'anthropic' : 'openai';
        this.send(res, 200, modelsListJson(wire, this.freeModelIds()), false);
        return;
      }
      if (req.method === 'POST' && (path === '/v1/chat/completions' || path === '/v1/messages')) {
        const api: ClientApi = path === '/v1/chat/completions' ? 'openai' : 'anthropic';
        const body = await this.readJsonBody(req);
        if (body === null) {
          const { status, body: errBody } = errorJson(api, 400, 'the request body is not a JSON object', 'invalid_request');
          this.send(res, status, errBody, false);
          return;
        }
        const result = await this.serveChat(api, body);
        this.send(res, result.status, result.payload, result.sse);
        return;
      }
      const { status, body: errBody } = errorJson('openai', 404, `no route for ${req.method} ${path}`, 'not_found');
      this.send(res, status, errBody, false);
    });
    this.queue = run.catch(() => undefined);
    run.catch((err) => {
      log(`request failed: ${(err as Error).message}`);
      try {
        this.send(res, 500, errorJson('openai', 500, 'the aggregator failed', 'internal_error').body, false);
      } catch {
        // The response is already gone; nothing more to do.
      }
    });
  }
}

/** Register with the env, retrying while it is not ready yet. */
async function register(port: number, catalog: Catalog): Promise<void> {
  const url = `${ENV_URL}/register`;
  for (let attempt = 1; attempt <= 50; attempt++) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: 1, port, catalog }),
        signal: AbortSignal.timeout(5000),
      });
      if (res.ok) {
        log(`registered on port ${port}`);
        return;
      }
      log(`register attempt ${attempt}: HTTP ${res.status}`);
    } catch (err) {
      log(`register attempt ${attempt}: ${(err as Error).message}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error('the env never accepted the registration');
}

/** Poll the env until it says the trial is done, then exit 0. */
async function pollUntilDone(server: ReturnType<typeof createServer>): Promise<void> {
  let consecutiveFailures = 0;
  for (;;) {
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(5000) });
      if (!res.ok) {
        consecutiveFailures += 1;
      } else {
        consecutiveFailures = 0;
        const body: unknown = await res.json();
        if (body !== null && typeof body === 'object' && (body as { done?: unknown }).done === true) {
          log('the env says the trial is done; exiting 0');
          server.close();
          process.exit(0);
        }
      }
    } catch {
      consecutiveFailures += 1;
    }
    if (consecutiveFailures >= 50) {
      log('the env has not answered 50 polls in a row; exiting 1');
      server.close();
      process.exit(1);
    }
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const loaded = await load();
  if (args.includes('--list')) {
    process.stdout.write(`${JSON.stringify(loaded.catalog)}\n`);
    return;
  }
  const aggregator = new Aggregator(loaded);
  const server = createServer((req, res) => aggregator.handle(req, res));
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server did not bind to a port');
  let port = address.port;
  if (port === 8080) {
    // Port 8080 belongs to the env; never serve on it.
    await new Promise<void>((resolve) => server.close(() => resolve()));
    await new Promise<void>((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '0.0.0.0', () => resolve());
    });
    const retry = server.address();
    if (retry === null || typeof retry === 'string') throw new Error('the server did not rebind to a port');
    port = retry.port;
  }
  log(`listening on 0.0.0.0:${port}`);
  await register(port, loaded.catalog);
  await pollUntilDone(server);
}

main().catch((err) => {
  log(`fatal: ${(err as Error).message}`);
  process.exit(1);
});
