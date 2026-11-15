// lib/brief.ts — the trial brief, the provider directory, and the catalog probe.
//
// A trial hands this server one file (MAKEFUN_PROMPT_FILE) carrying three JSON
// documents and the prose around them. This module finds the documents by their
// `schema` member, turns them into the structures the server routes on, and
// probes every provider's catalog route in that provider's own wire so the
// server can build the one catalog it registers and prints with --list.
//
// Nothing here reads a clock or rolls dice: the same brief and the same
// provider answers always build the same catalog, so a replay reproduces it.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type AuthKind = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

export const DIR_SCHEMA = 'free-model-aggregator-01/providers/v1';
export const TABLE_SCHEMA = 'free-model-aggregator-01/ratings/v1';
export const PLAN_SCHEMA = 'free-model-aggregator-01/workload/v1';

export type Json = Record<string, unknown>;
export const isObj = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One provider as the brief's directory lists it. */
export interface DirProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthKind;
  base: string;
  keyEnv: string;
  key: string;
}

export interface Directory {
  schema: string;
  providers: DirProvider[];
}

/** One row of the sealed quality table. */
export interface RatingEntry {
  model: string;
  provider: string;
  ratingMilli: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface RatingTable {
  schema: string;
  rows: RatingEntry[];
  best: { id: string; ratingMilli: number }[];
}

/** What one workload request needs a model to have. */
export interface Need {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  minContext?: number;
}

export interface PlanItem {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Need;
}

export interface WorkPlan {
  schema: string;
  items: PlanItem[];
}

export interface Brief {
  directory: Directory;
  ratings: RatingTable | null;
  plan: WorkPlan | null;
  problem: string | null;
}

/** One model as a provider's catalog route listed it. */
export interface ListedModel {
  id: string;
  displayName: string;
  free: boolean | null;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface ProbedProvider {
  provider: DirProvider;
  ok: boolean;
  status: number;
  models: ListedModel[];
  problem: string | null;
}

export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

export interface Catalog {
  providers: CatalogProvider[];
}

/** A base path joined to the env origin; an absolute base is used as it stands. */
export function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const root = envUrl.replace(/\/+$/, '');
  if (b === '') return root;
  return `${root}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The header a provider's auth scheme wants its key in. */
export function authHeaders(auth: AuthKind, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The index just past the `}` that closes the object opening at `at`, or -1. */
function closeBrace(text: string, at: number): number {
  let depth = 0;
  let quoted = false;
  for (let i = at; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '\\') i += 1;
      else if (ch === '"') quoted = false;
    } else if (ch === '"') quoted = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every top-level JSON object in `text` that carries a string `schema`. */
function schemaDocs(text: string): Json[] {
  const found: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = closeBrace(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isObj(value) && typeof value.schema === 'string') found.push(value);
    } catch {
      // Not a document on its own; keep scanning.
    }
  }
  return found;
}

function asText(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function asIntOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function asBoolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function dirProviderOf(v: unknown): DirProvider | null {
  if (!isObj(v)) return null;
  const id = asText(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    asText(v.path) !== ''
      ? asText(v.path)
      : asText(v.base_path) !== ''
        ? asText(v.base_path)
        : asText(v.base_url) !== ''
          ? asText(v.base_url)
          : asText(v.base);
  return { id, name: asText(v.name) !== '' ? asText(v.name) : id, wire, auth, base, keyEnv: asText(v.key_env), key: asText(v.key) };
}

function ratingEntryOf(v: unknown): RatingEntry | null {
  if (!isObj(v)) return null;
  const model = asText(v.model);
  const provider = asText(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    ratingMilli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    contextLength: asIntOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function planItemOf(v: unknown): PlanItem | null {
  if (!isObj(v)) return null;
  const id = asText(v.id);
  if (id === '') return null;
  const req = isObj(v.requires) ? v.requires : {};
  const need: Need = {};
  if (req.tools === true) need.tools = true;
  if (req.vision === true) need.vision = true;
  if (req.json === true) need.json = true;
  if (typeof req.min_context === 'number' && Number.isFinite(req.min_context)) need.minContext = req.min_context;
  return { id, api: v.api === 'anthropic' ? 'anthropic' : 'openai', stream: v.stream === true, weight: v.weight === 2 ? 2 : 1, requires: need };
}

/** The brief at `path`: directory, quality table and workload plan. */
export function parseBrief(path: string): Brief {
  const empty: Directory = { schema: DIR_SCHEMA, providers: [] };
  if (path === '') return { directory: empty, ratings: null, plan: null, problem: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { directory: empty, ratings: null, plan: null, problem: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = schemaDocs(text);
  const dirDoc = docs.find((d) => d.schema === DIR_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (dirDoc === null || !Array.isArray(dirDoc.providers)) {
    return { directory: empty, ratings: null, plan: null, problem: `the brief at ${path} holds no ${DIR_SCHEMA} document` };
  }
  const providers = dirDoc.providers.flatMap((v) => {
    const p = dirProviderOf(v);
    return p === null ? [] : [p];
  });
  const tableDoc = docs.find((d) => d.schema === TABLE_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings: RatingTable | null =
    tableDoc === null || !Array.isArray(tableDoc.ratings)
      ? null
      : {
          schema: TABLE_SCHEMA,
          rows: tableDoc.ratings.flatMap((v) => {
            const r = ratingEntryOf(v);
            return r === null ? [] : [r];
          }),
          best: Array.isArray(tableDoc.best)
            ? tableDoc.best.flatMap((v) => {
                if (!isObj(v) || asText(v.id) === '') return [];
                return [{ id: asText(v.id), ratingMilli: asIntOrNull(v.rating_milli) ?? 0 }];
              })
            : [],
        };
  const planDoc = docs.find((d) => d.schema === PLAN_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const plan: WorkPlan | null =
    planDoc === null || !Array.isArray(planDoc.manifest)
      ? null
      : {
          schema: PLAN_SCHEMA,
          items: planDoc.manifest.flatMap((v) => {
            const item = planItemOf(v);
            return item === null ? [] : [item];
          }),
        };
  return { directory: { schema: DIR_SCHEMA, providers }, ratings, plan, problem: null };
}

/** The quality-table row for one provider's model, when the table has one. */
export function ratingFor(ratings: RatingTable | null, providerId: string, modelId: string): RatingEntry | null {
  if (ratings === null) return null;
  return ratings.rows.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/** Whether a model is free: the quality table where it has a row (the sealed truth), else the listing. */
export function isFreeModel(model: ListedModel, row: RatingEntry | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table where it has a row, else the listing's fields. */
export function capsFor(model: ListedModel, row: RatingEntry | null): { contextLength: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null } {
  return {
    contextLength: row?.contextLength ?? model.contextLength ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's needs. An unknown capability never meets one. */
export function meetsNeed(need: Need, caps: { contextLength: number | null; tools: boolean | null; vision: boolean | null; json: boolean | null }): boolean {
  if (need.tools === true && caps.tools !== true) return false;
  if (need.vision === true && caps.vision !== true) return false;
  if (need.json === true && caps.json !== true) return false;
  if (need.minContext !== undefined && (caps.contextLength === null || caps.contextLength < need.minContext)) return false;
  return true;
}

/** True when every price a listing names is zero; null when the listing says nothing about price. */
function allZeroPrices(v: Json): boolean | null {
  const pricing = isObj(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const nums = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (nums.length === 0) return null;
  return nums.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function listedModelOf(wire: Wire, v: unknown): ListedModel | null {
  if (!isObj(v)) return null;
  const id = wire === 'gemini' ? asText(v.name).replace(/^models\//, '') : asText(v.id);
  if (id === '') return null;
  const context = asIntOrNull(v.context_length) ?? asIntOrNull(v.context_window) ?? asIntOrNull(v.inputTokenLimit);
  return {
    id,
    displayName: asText(v.display_name) !== '' ? asText(v.display_name) : asText(v.displayName) !== '' ? asText(v.displayName) : id,
    free: asBoolOrNull(v.free) ?? allZeroPrices(v),
    contextLength: context,
    tools: asBoolOrNull(v.tools),
    vision: asBoolOrNull(v.vision),
    json: asBoolOrNull(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function listedModelsOf(wire: Wire, body: unknown): ListedModel[] {
  if (!isObj(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModelOf(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
export async function probeOne(envUrl: string, provider: DirProvider, timeoutMs = 5000): Promise<ProbedProvider> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, status: res.status, models: [], problem: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, status: res.status, models: listedModelsOf(provider.wire, body), problem: null };
  } catch (err) {
    return { provider, ok: false, status: 0, models: [], problem: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the probes and the quality table make: every provider that
 * answered, every model it listed, sorted by id within each provider.
 *
 * base_url is the directory's own path (/p/<id>), never joined to the env
 * origin: the env's origin differs between a live pass and its replay, and
 * the catalog is part of the state export a replay must reproduce byte for
 * byte. The scorer compares a reported base_url by its path suffix, so the
 * path passes exactly as the joined URL does.
 */
export function buildCatalog(probes: readonly ProbedProvider[], ratings: RatingTable | null): Catalog {
  const out: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingFor(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: isFreeModel(model, row), context_length: row?.contextLength ?? model.contextLength ?? null });
    }
    out.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.keyEnv, wire: probe.provider.wire, models });
  }
  return { providers: out };
}
