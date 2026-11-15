// lib/translate.ts — the three provider wires and the two client wires.
//
// The client talks to this server in OpenAI or Anthropic shape, and this
// server talks to the providers in OpenAI, Anthropic or Gemini shape. A
// request therefore crosses two translations: the client's wire into the
// provider's wire, and the provider's answer back into the client's wire.
//
// The one rule that governs every translation: pass the request through
// unchanged in meaning. The provider echoes back a fingerprint of the
// request it received (system text, user text, tool names, max tokens, stop
// strings, image bytes, whether JSON mode was asked for) and the client
// compares that fingerprint with the request it sent, field by field. A
// field dropped, renamed or rewritten anywhere is a request the client
// refuses to count. So nothing here invents, drops or rewords a field —
// it only re-spells the same request in the other wire's vocabulary.
//
// The providers answer in pure text only (never a tool call), so the
// answer side is simpler: one text string, its token usage, and its stop
// reason, rendered into whichever wire the client asked for.
//
// All of it is deterministic: fixed response ids and timestamps, fixed
// chunking, no clock, no randomness.

import { createHash } from 'node:crypto';
import type { Json } from './brief.ts';
import { isObj } from './brief.ts';

export type ClientApi = 'openai' | 'anthropic';
export type ProviderWire = 'openai' | 'anthropic' | 'gemini';

export interface Usage {
  input: number;
  output: number;
}

/** One provider answer, reduced to the three facts a client checks. */
export interface ProviderAnswer {
  text: string;
  usage: Usage;
  /** The provider's stop reason, as it wrote it. */
  stop: string;
}

export const FIXED_CREATED = 1767225600;
export const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';

const STREAM_PIECE = 13;

function sha8(value: string): string {
  return createHash('sha256').update(value, 'utf8').digest('hex').slice(0, 8);
}

function sha8Text(value: string): string {
  return sha8(value);
}

/** Split a string into the 13-character pieces a provider streams in. */
export function textPieces(text: string): string[] {
  if (text === '') return [];
  const out: string[] = [];
  for (let i = 0; i < text.length; i += STREAM_PIECE) out.push(text.slice(i, i + STREAM_PIECE));
  return out;
}

/** The family of a stop reason: 0 stop, 1 length, 2 tool call, 3 refusal. */
export function stopFamily(reason: string): number {
  const r = reason.toLowerCase();
  if (r === 'stop' || r === 'end_turn' || r === 'stop_sequence' || r === 'stop_sequence') return 0;
  if (r === 'length' || r === 'max_tokens') return 1;
  if (r === 'tool_calls' || r === 'function_call' || r === 'tool_use') return 2;
  if (r === 'content_filter' || r === 'refusal' || r === 'safety') return 3;
  return 0;
}

/** The name of a stop family in one wire's vocabulary. */
export function stopName(wire: ClientApi, family: number): string {
  if (wire === 'openai') return ['stop', 'length', 'tool_calls', 'content_filter'][family] ?? 'stop';
  return ['end_turn', 'max_tokens', 'tool_use', 'refusal'][family] ?? 'end_turn';
}

// ---------------------------------------------------------------------------
// Reading the client's request
// ---------------------------------------------------------------------------

type Part = { type: 'text'; text: string } | { type: 'image'; base64: string | null; mediaType: string; url: string | null };

interface ChatMsg {
  role: 'system' | 'user' | 'assistant' | 'tool';
  text: string;
  parts: Part[];
  toolCallId: string | null;
  toolCalls: { id: string; name: string; input: Json }[];
}

/** The parts of one OpenAI message's content, as text and image parts. */
function openAIParts(content: unknown): Part[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const out: Part[] = [];
  for (const p of content) {
    if (!isObj(p)) continue;
    if (p.type === 'text' && typeof p.text === 'string') out.push({ type: 'text', text: p.text });
    if (p.type === 'image_url' && isObj(p.image_url) && typeof p.image_url.url === 'string') {
      const url = p.image_url.url;
      const m = url.match(/^data:([^;,]+);base64,(.*)$/s);
      if (m) out.push({ type: 'image', base64: m[2], mediaType: m[1], url: null });
      else out.push({ type: 'image', base64: null, mediaType: 'image/png', url });
    }
  }
  return out;
}

/** One OpenAI message reduced to its role, text and parts. */
function openAIMessage(raw: unknown): ChatMsg | null {
  if (!isObj(raw)) return null;
  const role = raw.role === 'system' || raw.role === 'user' || raw.role === 'assistant' || raw.role === 'tool' ? raw.role : null;
  if (role === null) return null;
  const parts = openAIParts(raw.content);
  const text = parts.filter((p) => p.type === 'text').map((p) => (p as { text: string }).text).join('');
  const toolCalls: ChatMsg['toolCalls'] = [];
  if (Array.isArray(raw.tool_calls)) {
    for (const tc of raw.tool_calls) {
      if (!isObj(tc) || !isObj(tc.function) || typeof tc.function.name !== 'string') continue;
      let input: Json = {};
      if (typeof tc.function.arguments === 'string') {
        try {
          const parsed: unknown = JSON.parse(tc.function.arguments);
          if (isObj(parsed)) input = parsed;
        } catch {
          input = {};
        }
      } else if (isObj(tc.function.arguments)) input = tc.function.arguments;
      toolCalls.push({ id: typeof tc.id === 'string' ? tc.id : 'call_1', name: tc.function.name, input });
    }
  }
  return { role, text, parts, toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : null, toolCalls };
}

/** The text of the parts of one Anthropic content block list, with images kept as parts. */
function anthropicParts(content: unknown): Part[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [];
  const out: Part[] = [];
  for (const b of content) {
    if (!isObj(b)) continue;
    if (b.type === 'text' && typeof b.text === 'string') out.push({ type: 'text', text: b.text });
    if (b.type === 'image' && isObj(b.source)) {
      if (b.source.type === 'base64' && typeof b.source.data === 'string') {
        out.push({ type: 'image', base64: b.source.data, mediaType: typeof b.source.media_type === 'string' ? b.source.media_type : 'image/png', url: null });
      } else if (b.source.type === 'url' && typeof b.source.url === 'string') {
        out.push({ type: 'image', base64: null, mediaType: 'image/png', url: b.source.url });
      }
    }
  }
  return out;
}

/** One Anthropic message reduced the same way an OpenAI message is. */
function anthropicMessage(raw: unknown): ChatMsg | null {
  if (!isObj(raw)) return null;
  const role = raw.role === 'user' || raw.role === 'assistant' ? raw.role : null;
  if (role === null) return null;
  const parts = anthropicParts(raw.content);
  const text = parts.filter((p) => p.type === 'text').map((p) => (p as { text: string }).text).join('');
  const toolCalls: ChatMsg['toolCalls'] = [];
  if (Array.isArray(raw.content)) {
    for (const b of raw.content) {
      if (isObj(b) && b.type === 'tool_use' && typeof b.name === 'string') {
        toolCalls.push({ id: typeof b.id === 'string' ? b.id : 'toolu_1', name: b.name, input: isObj(b.input) ? b.input : {} });
      }
    }
  }
  return { role, text, parts, toolCallId: null, toolCalls };
}

/** The messages of a client body, whichever wire it arrived in. */
export function clientMessages(api: ClientApi, body: Json): ChatMsg[] {
  if (!Array.isArray(body.messages)) return [];
  const out: ChatMsg[] = [];
  for (const raw of body.messages) {
    const m = api === 'openai' ? openAIMessage(raw) : anthropicMessage(raw);
    if (m !== null) out.push(m);
  }
  return out;
}

/** The system text of a client body: a system message, or Anthropic's system field. */
export function clientSystem(api: ClientApi, body: Json): string {
  const messages = clientMessages(api, body);
  const fromMessages = messages.filter((m) => m.role === 'system').map((m) => m.text).join('\n');
  if (fromMessages !== '') return fromMessages;
  return typeof body.system === 'string' ? body.system : '';
}

/** The user text of the last user message, for the echo's fingerprint. */
export function clientUserText(api: ClientApi, body: Json): string {
  const messages = clientMessages(api, body).filter((m) => m.role === 'user');
  return messages.length === 0 ? '' : messages[messages.length - 1].text;
}

/** The tool names the client asked for, in order. */
export function clientToolNames(api: ClientApi, body: Json): string[] {
  if (!Array.isArray(body.tools)) return [];
  return body.tools.flatMap((t) => {
    if (!isObj(t)) return [];
    if (api === 'openai' && isObj(t.function) && typeof t.function.name === 'string') return [t.function.name];
    if (api === 'anthropic' && typeof t.name === 'string') return [t.name];
    return [];
  });
}

/** Whether the client's request asked for JSON mode. Only OpenAI can ask for it. */
export function clientJsonMode(api: ClientApi, body: Json): boolean {
  if (api !== 'openai') return false;
  if (!isObj(body.response_format)) return false;
  return body.response_format.type === 'json_object' || body.response_format.type === 'json_schema';
}

/** The stop strings the client asked for, whichever wire it arrived in. */
export function clientStops(api: ClientApi, body: Json): string[] {
  if (api === 'openai' && Array.isArray(body.stop)) return body.stop.filter((s): s is string => typeof s === 'string');
  if (api === 'anthropic' && Array.isArray(body.stop_sequences)) return body.stop_sequences.filter((s): s is string => typeof s === 'string');
  return [];
}

/** The max tokens the client asked for (1024 when it did not say). */
export function clientMaxTokens(body: Json): number {
  if (typeof body.max_tokens === 'number' && Number.isFinite(body.max_tokens)) return body.max_tokens;
  if (typeof body.max_completion_tokens === 'number' && Number.isFinite(body.max_completion_tokens)) return body.max_completion_tokens;
  return 1024;
}

/** The image fingerprints of a client body: sha8 of each image's bytes. */
export function clientImageShas(api: ClientApi, body: Json): string[] {
  const out: string[] = [];
  for (const m of clientMessages(api, body)) {
    for (const p of m.parts) {
      if (p.type !== 'image') continue;
      if (p.base64 !== null) out.push(sha8(Buffer.from(p.base64, 'base64').toString('utf8')));
      else if (p.url !== null) out.push(sha8Text(p.url));
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// The client wire, translated into a provider's wire
// ---------------------------------------------------------------------------

/** An OpenAI message body for a user/assistant message, from its parts. */
function openAIContentOf(parts: Part[]): unknown {
  const blocks: unknown[] = [];
  for (const p of parts) {
    if (p.type === 'text') blocks.push({ type: 'text', text: p.text });
    else if (p.base64 !== null) blocks.push({ type: 'image_url', image_url: { url: `data:${p.mediaType};base64,${p.base64}` } });
    else if (p.url !== null) blocks.push({ type: 'image_url', image_url: { url: p.url } });
  }
  if (blocks.length === 1 && isObj(blocks[0]) && blocks[0].type === 'text') return (blocks[0] as { text: string }).text;
  return blocks;
}

/** An Anthropic content block list for a message, from its parts. */
function anthropicContentOf(parts: Part[], toolCalls: ChatMsg['toolCalls']): unknown[] {
  const blocks: unknown[] = [];
  for (const p of parts) {
    if (p.type === 'text') blocks.push({ type: 'text', text: p.text });
    else if (p.base64 !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: p.mediaType, data: p.base64 } });
    else if (p.url !== null) blocks.push({ type: 'image', source: { type: 'url', url: p.url } });
  }
  for (const tc of toolCalls) blocks.push({ type: 'tool_use', id: tc.id, name: tc.name, input: tc.input });
  return blocks;
}

/** A Gemini content part list for a message, from its parts. */
function geminiPartsOf(parts: Part[], toolCalls: ChatMsg['toolCalls']): unknown[] {
  const out: unknown[] = [];
  for (const p of parts) {
    if (p.type === 'text') out.push({ text: p.text });
    else if (p.base64 !== null) out.push({ inlineData: { mimeType: p.mediaType, data: p.base64 } });
    else if (p.url !== null) out.push({ fileData: { fileUri: p.url } });
  }
  for (const tc of toolCalls) out.push({ functionCall: { name: tc.name, args: tc.input } });
  return out;
}

function openAIToolsOf(api: ClientApi, body: Json): unknown[] | null {
  if (!Array.isArray(body.tools)) return null;
  return body.tools.flatMap((t) => {
    if (!isObj(t)) return [];
    if (api === 'openai' && isObj(t.function) && typeof t.function.name === 'string') {
      return [{ type: 'function', function: { name: t.function.name, description: typeof t.function.description === 'string' ? t.function.description : undefined, parameters: isObj(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } } }];
    }
    if (api === 'anthropic' && typeof t.name === 'string') {
      return [{ type: 'function', function: { name: t.name, description: typeof t.description === 'string' ? t.description : undefined, parameters: isObj(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } } }];
    }
    return [];
  });
}

function anthropicToolsOf(api: ClientApi, body: Json): unknown[] | null {
  if (!Array.isArray(body.tools)) return null;
  return body.tools.flatMap((t) => {
    if (!isObj(t)) return [];
    if (api === 'openai' && isObj(t.function) && typeof t.function.name === 'string') {
      return [{ name: t.function.name, description: typeof t.function.description === 'string' ? t.function.description : '', input_schema: isObj(t.function.parameters) ? t.function.parameters : { type: 'object', properties: {} } }];
    }
    if (api === 'anthropic' && typeof t.name === 'string') {
      return [{ name: t.name, description: typeof t.description === 'string' ? t.description : '', input_schema: isObj(t.input_schema) ? t.input_schema : { type: 'object', properties: {} } }];
    }
    return [];
  });
}

function geminiToolsOf(api: ClientApi, body: Json): unknown[] | null {
  if (!Array.isArray(body.tools)) return null;
  const decls = body.tools.flatMap((t) => {
    if (!isObj(t)) return [];
    if (api === 'openai' && isObj(t.function) && typeof t.function.name === 'string') {
      return [{ name: t.function.name, description: typeof t.function.description === 'string' ? t.function.description : '', parameters: isObj(t.function.parameters) ? t.function.parameters : undefined }];
    }
    if (api === 'anthropic' && typeof t.name === 'string') {
      return [{ name: t.name, description: typeof t.description === 'string' ? t.description : '', parameters: isObj(t.input_schema) ? t.input_schema : undefined }];
    }
    return [];
  });
  return decls.length === 0 ? null : [{ functionDeclarations: decls }];
}

function openAIToolChoiceOf(api: ClientApi, body: Json): unknown {
  if (!isObj(body.tool_choice)) return undefined;
  const tc = body.tool_choice;
  if (api === 'anthropic') {
    if (tc.type === 'auto') return 'auto';
    if (tc.type === 'any') return 'required';
    if (tc.type === 'none') return 'none';
    if (tc.type === 'tool' && typeof tc.name === 'string') return { type: 'function', function: { name: tc.name } };
    return undefined;
  }
  return tc;
}

function anthropicToolChoiceOf(api: ClientApi, body: Json): unknown {
  if (!isObj(body.tool_choice)) return undefined;
  const tc = body.tool_choice;
  if (api === 'openai') {
    if (typeof tc === 'string') {
      if (tc === 'auto') return { type: 'auto' };
      if (tc === 'required') return { type: 'any' };
      if (tc === 'none') return { type: 'none' };
      return undefined;
    }
    if (tc.type === 'function' && isObj(tc.function) && typeof tc.function.name === 'string') return { type: 'tool', name: tc.function.name };
    return undefined;
  }
  return tc;
}

/**
 * The upstream OpenAI request body for a client request.
 * When the client already speaks OpenAI this is a faithful copy with the
 * model and streaming set; when it speaks Anthropic its messages, tools and
 * tool choice are re-spelled, and no response_format is ever added (the
 * client's Anthropic echo expects json_mode to stay false).
 */
export function openAIUpstream(api: ClientApi, body: Json, model: string, stream: boolean): Json {
  const out: Json = { model, stream };
  if (api === 'openai') {
    const messages = clientMessages('openai', body);
    out.messages = messages.map((m) => {
      if (m.role === 'tool') return { role: 'tool', tool_call_id: m.toolCallId ?? 'call_1', content: m.text };
      const base: Json = { role: m.role, content: openAIContentOf(m.parts) };
      if (m.role === 'assistant' && m.toolCalls.length > 0) {
        base.tool_calls = m.toolCalls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: JSON.stringify(tc.input) } }));
        if (m.text === '') base.content = null;
      }
      return base;
    });
    const tools = openAIToolsOf('openai', body);
    if (tools !== null && tools.length > 0) out.tools = tools;
    const choice = openAIToolChoiceOf('openai', body);
    if (choice !== undefined) out.tool_choice = choice;
    out.max_tokens = clientMaxTokens(body);
    if (typeof body.temperature === 'number') out.temperature = body.temperature;
    if (typeof body.top_p === 'number') out.top_p = body.top_p;
    const stops = clientStops('openai', body);
    if (stops.length > 0) out.stop = stops;
    if (isObj(body.response_format)) out.response_format = body.response_format;
  } else {
    const system = clientSystem('anthropic', body);
    const messages = clientMessages('anthropic', body);
    const toolResults: { tool_call_id: string; content: string }[] = [];
    const outMessages: unknown[] = [];
    for (const m of messages) {
      if (m.role === 'user') {
        const blocks = openAIContentOf(m.parts);
        outMessages.push({ role: 'user', content: blocks });
      } else if (m.role === 'assistant') {
        const base: Json = { role: 'assistant', content: m.text === '' && m.toolCalls.length > 0 ? null : openAIContentOf(m.parts) };
        if (m.toolCalls.length > 0) {
          base.tool_calls = m.toolCalls.map((tc) => ({ id: tc.id, type: 'function', function: { name: tc.name, arguments: JSON.stringify(tc.input) } }));
        }
        outMessages.push(base);
      }
    }
    if (system !== '') outMessages.unshift({ role: 'system', content: system });
    for (const tr of toolResults) outMessages.push({ role: 'tool', tool_call_id: tr.tool_call_id, content: tr.content });
    out.messages = outMessages;
    const tools = openAIToolsOf('anthropic', body);
    if (tools !== null && tools.length > 0) out.tools = tools;
    const choice = openAIToolChoiceOf('anthropic', body);
    if (choice !== undefined) out.tool_choice = choice;
    out.max_tokens = clientMaxTokens(body);
    if (typeof body.temperature === 'number') out.temperature = body.temperature;
    if (typeof body.top_p === 'number') out.top_p = body.top_p;
    const stops = clientStops('anthropic', body);
    if (stops.length > 0) out.stop = stops;
  }
  if (stream) out.stream_options = { include_usage: true };
  return out;
}

/**
 * The upstream Anthropic request body for a client request.
 * An OpenAI system message becomes the system field; tool results are
 * folded into the next user message so the user fingerprint stays the
 * request's own (a lone tool-result user message would wipe it to "").
 */
export function anthropicUpstream(api: ClientApi, body: Json, model: string, stream: boolean): Json {
  const out: Json = { model, stream, max_tokens: clientMaxTokens(body) };
  const system = clientSystem(api, body);
  if (system !== '') out.system = system;
  const messages = clientMessages(api, body);
  const outMessages: Json[] = [];
  let pendingToolResults: { tool_use_id: string; content: string }[] = [];
  const flushToolResults = (into: Json) => {
    if (pendingToolResults.length === 0) return;
    const blocks = Array.isArray(into.content) ? (into.content as unknown[]) : [];
    for (const tr of pendingToolResults) blocks.push({ type: 'tool_result', tool_use_id: tr.tool_use_id, content: tr.content });
    into.content = blocks;
    pendingToolResults = [];
  };
  for (const m of messages) {
    if (m.role === 'system') continue;
    if (m.role === 'tool') {
      pendingToolResults.push({ tool_use_id: m.toolCallId ?? 'call_1', content: m.text });
      continue;
    }
    if (m.role === 'user') {
      const msg: Json = { role: 'user', content: anthropicContentOf(m.parts, []) };
      flushToolResults(msg);
      outMessages.push(msg);
    } else if (m.role === 'assistant') {
      const msg: Json = { role: 'assistant', content: anthropicContentOf(m.parts, m.toolCalls) };
      outMessages.push(msg);
    }
  }
  if (pendingToolResults.length > 0 && outMessages.length > 0) {
    const last = outMessages[outMessages.length - 1];
    if (last.role === 'user') flushToolResults(last);
    else if (last.role === 'assistant') {
      // A tool result has no user message left to join; keep it honest as a
      // user message of tool results, which the parser reads as empty text.
      outMessages.push({ role: 'user', content: pendingToolResults.map((tr) => ({ type: 'tool_result', tool_use_id: tr.tool_use_id, content: tr.content })) });
    }
    pendingToolResults = [];
  }
  out.messages = outMessages;
  const tools = anthropicToolsOf(api, body);
  if (tools !== null && tools.length > 0) out.tools = tools;
  const choice = anthropicToolChoiceOf(api, body);
  if (choice !== undefined) out.tool_choice = choice;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = clientStops(api, body);
  if (stops.length > 0) out.stop_sequences = stops;
  return out;
}

/**
 * The upstream Gemini request body for a client request.
 * systemInstruction carries the system text; contents carry the turns;
 * generationConfig carries max tokens, stops and — only when the OpenAI
 * client asked for JSON mode — the JSON mime type. An Anthropic client
 * never gets a mime type, because its echo expects json_mode to stay false.
 */
export function geminiUpstream(api: ClientApi, body: Json, model: string, stream: boolean): Json {
  void model;
  void stream;
  const out: Json = {};
  const system = clientSystem(api, body);
  if (system !== '') out.systemInstruction = { parts: [{ text: system }] };
  const messages = clientMessages(api, body);
  const contents: unknown[] = [];
  for (const m of messages) {
    if (m.role === 'system' || m.role === 'tool') continue;
    if (m.role === 'user') contents.push({ role: 'user', parts: geminiPartsOf(m.parts, []) });
    else if (m.role === 'assistant') contents.push({ role: 'model', parts: geminiPartsOf(m.parts, m.toolCalls) });
  }
  out.contents = contents;
  const tools = geminiToolsOf(api, body);
  if (tools !== null) out.tools = tools;
  const config: Json = { maxOutputTokens: clientMaxTokens(body) };
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const stops = clientStops(api, body);
  if (stops.length > 0) config.stopSequences = stops;
  if (api === 'openai' && clientJsonMode('openai', body)) config.responseMimeType = 'application/json';
  out.generationConfig = config;
  return out;
}

/** The upstream body for one provider's wire, from one client request. */
export function upstreamBody(api: ClientApi, wire: ProviderWire, body: Json, model: string, stream: boolean): Json {
  if (wire === 'openai') return openAIUpstream(api, body, model, stream);
  if (wire === 'anthropic') return anthropicUpstream(api, body, model, stream);
  return geminiUpstream(api, body, model, stream);
}

// ---------------------------------------------------------------------------
// Reading a provider's answer
// ---------------------------------------------------------------------------

/** One frame of a server-sent-events text: its event name and data lines. */
function sseFrames(text: string): { event: string; data: string }[] {
  const frames: { event: string; data: string }[] = [];
  const chunks = text.split(/\r?\n\r?\n/);
  for (const chunk of chunks) {
    let event = '';
    const datas: string[] = [];
    for (const line of chunk.split(/\r?\n/)) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) datas.push(line.slice('data:'.length).trimStart());
      else if (line.startsWith(':')) continue;
    }
    if (datas.length > 0) frames.push({ event, data: datas.join('\n') });
  }
  return frames;
}

function parseJsonSafe(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function usageOf(input: unknown, output: unknown): Usage {
  const i = typeof input === 'number' && Number.isFinite(input) ? input : 0;
  const o = typeof output === 'number' && Number.isFinite(output) ? output : 0;
  return { input: i, output: o };
}

/** An OpenAI provider's answer, JSON or streamed, reduced to text, usage and stop. */
export function parseOpenAIAnswer(text: string): ProviderAnswer {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) {
    const body: unknown = parseJsonSafe(trimmed);
    if (isObj(body) && Array.isArray(body.choices) && body.choices.length > 0) {
      const choice = isObj(body.choices[0]) ? body.choices[0] : {};
      const message = isObj(choice.message) ? choice.message : {};
      const delta = isObj(choice.delta) ? choice.delta : {};
      const content = typeof message.content === 'string' ? message.content : typeof delta.content === 'string' ? delta.content : '';
      const usage = isObj(body.usage) ? usageOf(body.usage.prompt_tokens, body.usage.completion_tokens) : { input: 0, output: 0 };
      return { text: content, usage, stop: typeof choice.finish_reason === 'string' ? choice.finish_reason : 'stop' };
    }
    return { text: '', usage: { input: 0, output: 0 }, stop: 'stop' };
  }
  let content = '';
  let usage: Usage = { input: 0, output: 0 };
  let stop = 'stop';
  for (const frame of sseFrames(text)) {
    if (frame.data === '[DONE]') break;
    const chunk: unknown = parseJsonSafe(frame.data);
    if (!isObj(chunk)) continue;
    if (Array.isArray(chunk.choices) && chunk.choices.length > 0) {
      const choice = isObj(chunk.choices[0]) ? chunk.choices[0] : {};
      const delta = isObj(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') content += delta.content;
      if (typeof choice.finish_reason === 'string') stop = choice.finish_reason;
    }
    if (isObj(chunk.usage)) usage = usageOf(chunk.usage.prompt_tokens, chunk.usage.completion_tokens);
  }
  return { text: content, usage, stop };
}

/** An Anthropic provider's answer, JSON or streamed, reduced the same way. */
export function parseAnthropicAnswer(text: string): ProviderAnswer {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) {
    const body: unknown = parseJsonSafe(trimmed);
    if (isObj(body) && Array.isArray(body.content)) {
      const first = isObj(body.content[0]) ? body.content[0] : {};
      const usage = isObj(body.usage) ? usageOf(body.usage.input_tokens, body.usage.output_tokens) : { input: 0, output: 0 };
      return { text: typeof first.text === 'string' ? first.text : '', usage, stop: typeof body.stop_reason === 'string' ? body.stop_reason : 'end_turn' };
    }
    return { text: '', usage: { input: 0, output: 0 }, stop: 'end_turn' };
  }
  let content = '';
  let usage: Usage = { input: 0, output: 0 };
  let stop = 'end_turn';
  for (const frame of sseFrames(text)) {
    const event: unknown = parseJsonSafe(frame.data);
    if (!isObj(event)) continue;
    if (event.type === 'message_start' && isObj(event.message) && isObj(event.message.usage)) {
      usage.input = usageOf(event.message.usage.input_tokens, 0).input;
    }
    if (event.type === 'content_block_delta' && isObj(event.delta) && typeof event.delta.text === 'string') content += event.delta.text;
    if (event.type === 'message_delta') {
      if (isObj(event.delta) && typeof event.delta.stop_reason === 'string') stop = event.delta.stop_reason;
      if (isObj(event.usage)) usage.output = usageOf(0, event.usage.output_tokens).output;
    }
  }
  return { text: content, usage, stop };
}

/** A Gemini provider's answer, JSON or streamed, reduced the same way. */
export function parseGeminiAnswer(text: string): ProviderAnswer {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) {
    const body: unknown = parseJsonSafe(trimmed);
    if (isObj(body) && Array.isArray(body.candidates) && body.candidates.length > 0) {
      const cand = isObj(body.candidates[0]) ? body.candidates[0] : {};
      const content = isObj(cand.content) ? cand.content : {};
      const parts = Array.isArray(content.parts) ? content.parts : [];
      const first = isObj(parts[0]) ? parts[0] : {};
      const usage = isObj(body.usageMetadata) ? usageOf(body.usageMetadata.promptTokenCount, body.usageMetadata.candidatesTokenCount) : { input: 0, output: 0 };
      return { text: typeof first.text === 'string' ? first.text : '', usage, stop: typeof cand.finishReason === 'string' ? cand.finishReason : 'STOP' };
    }
    return { text: '', usage: { input: 0, output: 0 }, stop: 'STOP' };
  }
  let content = '';
  let usage: Usage = { input: 0, output: 0 };
  let stop = 'STOP';
  for (const frame of sseFrames(text)) {
    const chunk: unknown = parseJsonSafe(frame.data);
    if (!isObj(chunk) || !Array.isArray(chunk.candidates) || chunk.candidates.length === 0) continue;
    const cand = isObj(chunk.candidates[0]) ? chunk.candidates[0] : {};
    const c = isObj(cand.content) ? cand.content : {};
    const parts = Array.isArray(c.parts) ? c.parts : [];
    for (const p of parts) if (isObj(p) && typeof p.text === 'string') content += p.text;
    if (typeof cand.finishReason === 'string') stop = cand.finishReason;
    if (isObj(chunk.usageMetadata)) usage = usageOf(chunk.usageMetadata.promptTokenCount, chunk.usageMetadata.candidatesTokenCount);
  }
  return { text: content, usage, stop };
}

/** A provider's answer in its own wire. */
export function parseProviderAnswer(wire: ProviderWire, text: string): ProviderAnswer {
  if (wire === 'openai') return parseOpenAIAnswer(text);
  if (wire === 'anthropic') return parseAnthropicAnswer(text);
  return parseGeminiAnswer(text);
}

// ---------------------------------------------------------------------------
// Rendering a provider's answer in the client's wire
// ---------------------------------------------------------------------------

/** One OpenAI chat-completion JSON body, carrying a provider's answer. */
export function renderOpenAIJson(model: string, answer: ProviderAnswer): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: answer.text }, finish_reason: stopName('openai', stopFamily(answer.stop)), logprobs: null }],
    usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output },
  };
}

/** One OpenAI chat-completion stream: role chunk, content pieces, finish, usage, [DONE]. */
export function renderOpenAIStream(model: string, answer: ProviderAnswer): string {
  const base = { id: 'chatcmpl-agg', object: 'chat.completion.chunk', created: FIXED_CREATED, model };
  const frames: string[] = [];
  const push = (chunk: unknown) => {
    frames.push(`data: ${JSON.stringify(chunk)}\n\n`);
  };
  push({ ...base, choices: [{ index: 0, delta: { role: 'assistant', content: '' }, finish_reason: null }] });
  for (const piece of textPieces(answer.text)) {
    push({ ...base, choices: [{ index: 0, delta: { content: piece }, finish_reason: null }] });
  }
  push({ ...base, choices: [{ index: 0, delta: {}, finish_reason: stopName('openai', stopFamily(answer.stop)) }] });
  push({ ...base, choices: [], usage: { prompt_tokens: answer.usage.input, completion_tokens: answer.usage.output, total_tokens: answer.usage.input + answer.usage.output } });
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** One Anthropic message JSON body, carrying a provider's answer. */
export function renderAnthropicJson(model: string, answer: ProviderAnswer): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: answer.text }],
    stop_reason: stopName('anthropic', stopFamily(answer.stop)),
    stop_sequence: null,
    usage: { input_tokens: answer.usage.input, output_tokens: answer.usage.output },
  };
}

/** One Anthropic message stream, in the event order the wire requires. */
export function renderAnthropicStream(model: string, answer: ProviderAnswer): string {
  const frames: string[] = [];
  const push = (event: string, data: unknown) => {
    frames.push(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };
  push('message_start', { type: 'message_start', message: { id: 'msg_agg', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: answer.usage.input, output_tokens: 0 } } });
  push('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } });
  for (const piece of textPieces(answer.text)) {
    push('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } });
  }
  push('content_block_stop', { type: 'content_block_stop', index: 0 });
  push('message_delta', { type: 'message_delta', delta: { stop_reason: stopName('anthropic', stopFamily(answer.stop)), stop_sequence: null }, usage: { output_tokens: answer.usage.output } });
  push('message_stop', { type: 'message_stop' });
  return frames.join('');
}

/** The GET /v1/models answer in the wire its auth header named. */
export function modelsListJson(wire: ClientApi, modelIds: string[]): Json {
  const ids = [...modelIds].sort();
  if (wire === 'openai') {
    return { object: 'list', data: ids.map((id) => ({ id, object: 'model', created: FIXED_CREATED, owned_by: 'free-model-aggregator' })) };
  }
  return { data: ids.map((id) => ({ type: 'model', id, display_name: id, created_at: FIXED_CREATED_AT })), has_more: false };
}

/** An error body in the client's wire. */
export function errorJson(wire: ClientApi, status: number, message: string, code: string): { status: number; body: Json } {
  if (wire === 'openai') {
    return { status, body: { error: { message, type: 'server_error', code } } };
  }
  return { status, body: { type: 'error', error: { type: 'api_error', message } } };
}
