// src/index.ts
import "dotenv/config";
import SlackBolt from "@slack/bolt";
import * as tencentcloud from "tencentcloud-sdk-nodejs-ocr";
import { z } from "zod";

/* ----------------------------- env + logger ----------------------------- */

const envSchema = z.object({
  PORT: z.coerce.number().default(3000),
  SLACK_SIGNING_SECRET: z.string().min(1),
  SLACK_BOT_TOKEN: z.string().min(1),
  TENCENTCLOUD_SECRET_ID: z.string().min(1),
  TENCENTCLOUD_SECRET_KEY: z.string().min(1),
  TENCENTCLOUD_REGION: z.string().min(1).default("ap-guangzhou"),
  LLM_API_KEY: z.string().min(1),
  LLM_API_URL: z.string().min(1).default("https://api.qnaigc.com/v1/chat/completions"),
  LLM_MODEL: z.string().min(1).default("deepseek/deepseek-v3.2-251201"),
});

const env = envSchema.parse(process.env);

function log(level: "INFO" | "WARN" | "ERROR", msg: string, extra: Record<string, any> = {}) {
  const payload = {
    ts: new Date().toISOString(),
    level,
    msg,
    ...extra,
  };
  if (level === "ERROR") console.error(JSON.stringify(payload));
  else if (level === "WARN") console.warn(JSON.stringify(payload));
  else console.log(JSON.stringify(payload));
}

const logInfo = (msg: string, extra: Record<string, any> = {}) => log("INFO", msg, extra);
const logWarn = (msg: string, extra: Record<string, any> = {}) => log("WARN", msg, extra);
const logError = (msg: string, extra: Record<string, any> = {}) => log("ERROR", msg, extra);

process.on("unhandledRejection", (reason: any) => {
  logError("process.unhandledRejection", { reason: String(reason), stack: reason?.stack });
});
process.on("uncaughtException", (err: any) => {
  logError("process.uncaughtException", { message: err?.message, stack: err?.stack });
});

/* ------------------------------ Slack Bolt ------------------------------ */

// Use ExpressReceiver for custom endpoints (/slack/events /slack/commands /slack/interactivity)
const { App, ExpressReceiver } = SlackBolt as typeof SlackBolt & {
  App: typeof SlackBolt.App;
  ExpressReceiver: typeof SlackBolt.ExpressReceiver;
};

const receiver = new ExpressReceiver({
  signingSecret: env.SLACK_SIGNING_SECRET,
  endpoints: {
    events: "/slack/events",
    commands: "/slack/commands",
    interactive: "/slack/interactivity",
  },
});

receiver.app.get("/health", (_req, res) => {
  res.status(200).send("ok");
});

const app = new App({
  token: env.SLACK_BOT_TOKEN,
  receiver,
});

/* ------------------------------ Tencent OCR ----------------------------- */

const OcrClient = tencentcloud.ocr.v20181119.Client;
const ocrClient = new OcrClient({
  credential: {
    secretId: env.TENCENTCLOUD_SECRET_ID,
    secretKey: env.TENCENTCLOUD_SECRET_KEY,
  },
  region: env.TENCENTCLOUD_REGION,
  profile: {
    httpProfile: {
      endpoint: "ocr.tencentcloudapi.com",
      // Optional: set request timeout for the SDK if supported by your version
      // reqTimeout: 10, // seconds (some SDK versions support this)
    },
  },
});

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

function isRetryableNetworkError(err: any) {
  const msg = String(err?.message || "");
  return (
    msg.includes("EAI_AGAIN") ||
    msg.includes("ENOTFOUND") ||
    msg.includes("ETIMEDOUT") ||
    msg.includes("ECONNRESET") ||
    msg.includes("socket hang up")
  );
}

async function fetchSlackFileAsBase64(fileUrl: string, ctx: Record<string, any>): Promise<string> {
  const start = Date.now();
  logInfo("slack.file.download.start", { ...ctx, fileUrl });

  const res = await fetch(fileUrl, {
    headers: {
      Authorization: `Bearer ${env.SLACK_BOT_TOKEN}`,
    },
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    logError("slack.file.download.failed", {
      ...ctx,
      status: res.status,
      ms: Date.now() - start,
      body_preview: body.slice(0, 300),
    });
    throw new Error(`Failed to download file: ${res.status}`);
  }

  const arrayBuffer = await res.arrayBuffer();
  logInfo("slack.file.download.ok", { ...ctx, ms: Date.now() - start, bytes: arrayBuffer.byteLength });
  return Buffer.from(arrayBuffer).toString("base64");
}

async function callTencentOCRWithSDK(imageBase64: string, ctx: Record<string, any>): Promise<string> {
  const maxAttempts = 4;

  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const start = Date.now();
    try {
      logInfo("tencent.ocr.request.start", { ...ctx, attempt });

      const result = await ocrClient.GeneralAccurateOCR({ ImageBase64: imageBase64 });

      const detCount = (result.TextDetections || []).length;
      logInfo("tencent.ocr.request.ok", { ...ctx, attempt, ms: Date.now() - start, detections: detCount });

      const dets = (result.TextDetections || []) as Array<{ DetectedText?: string }>;
      return dets
        .map((x) => x.DetectedText)
        .filter((x) => x && x.trim().length > 0)
        .join("\n");
    } catch (err: any) {
      logError("tencent.ocr.request.failed", {
        ...ctx,
        attempt,
        ms: Date.now() - start,
        message: err?.message,
        requestId: err?.requestId || "",
        traceId: err?.traceId || "",
      });

      if (attempt < maxAttempts && isRetryableNetworkError(err)) {
        const backoff = 300 * Math.pow(2, attempt - 1);
        logWarn("tencent.ocr.request.retrying", { ...ctx, attempt, backoff_ms: backoff });
        await sleep(backoff);
        continue;
      }

      throw new Error(`OCR Failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  return "";
}

/* --------------------------------- LLM --------------------------------- */

async function callLLMToAnalyze(
  ocrText: string,
  ctx: Record<string, any>,
): Promise<{ text: string; usage?: { prompt_tokens: number; completion_tokens: number } }> {
  const today = new Date().toISOString().split("T")[0];
  const systemPrompt = `
You are an expert OCR Data Extraction Auditor. Your goal is to extract precise structured data from receipts/invoices.

**Current Server Date:** ${today} (YYYY-MM-DD)
*Use this date to infer the year if missing, or to validate that the transaction date is not in the distant future.*

**Critical Instruction: Context Analysis**
Before extracting the final data, you MUST perform a "Context Analysis" to resolve ambiguities.

**Step 1: Infer Region & Country**
Analyze currency symbols, phone codes, addresses, and language.
- Japan (JP): Look for Yen symbol (¥), Katakana/Hiragana/Kanji, or "+81".
- China (CN): Look for Simplified Chinese, "+86".
- Korea (KR): Look for Hangul, Won symbol (₩), "+82".

**Step 2: Extract Data Based on Region (Date Format Rules)**
- **Store Name**: Look for the most prominent text header.
- **Date**: Extract and convert strictly to **YYYY-MM-DD**.
  - **China (CN) / Japan (JP) / Korea (KR)**:
    - The format is STRICTLY **Year-Month-Day** (Big-Endian).
    - **CRITICAL RULE**: If you see a format like "XX/XX/XX" (e.g., "26/01/22") in these regions, the **FIRST** number is the YEAR.
    - *Example*: "26/01/22" in Japan = 2026-01-22. (Do NOT interpret as 22nd Jan 2026).
  - **Singapore (SG) / UK / Hong Kong (HK)**:
    - The format is usually **Day-Month-Year** (Little-Endian).
    - *Example*: "26/01/22" in UK = 2022-01-26.
  - **USA (US)**:
    - The format is usually **Month-Day-Year** (Middle-Endian).

- **Items**: Summarize key purchases.
- **Total**: Amount + Currency Code (ISO 4217, e.g., SGD, HKD, USD, JPY, CNY).

**Step 3: Final Output Format**
Output ONLY the final result in the following clean Markdown format (No JSON, No introductory text):

*Receipt Summary*
*Store*: [Store Name]
*Country*: [Country Code]
*Date*: [YYYY-MM-DD]
-------------------
[Item Name]   [Price]
...
-------------------
*Total*: [Currency] [Amount]
`;

  const start = Date.now();
  logInfo("llm.analyze.request.start", { ...ctx, ocr_chars: ocrText.length });

  try {
    const res = await fetch(env.LLM_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.LLM_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        stream: false,
        model: env.LLM_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          {
            role: "user",
            content: `Analyze this raw OCR text:\n\n${ocrText}\n\nRemember: Infer the region first to decide if date is DD/MM or MM/DD.`,
          },
        ],
        temperature: 0.1,
      }),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) {
      logError("llm.analyze.request.failed", {
        ...ctx,
        ms: Date.now() - start,
        status: res.status,
        error: data?.error,
      });
      throw new Error(data?.error?.message || `LLM API error (status ${res.status})`);
    }

    const content = data.choices?.[0]?.message?.content;
    const usage = data.usage;

    logInfo("llm.analyze.request.ok", { ...ctx, ms: Date.now() - start, usage });

    return {
      text: content || "⚠️ AI could not analyze the text.",
      usage,
    };
  } catch (error: any) {
    logError("llm.analyze.request.exception", { ...ctx, ms: Date.now() - start, message: error?.message });
    return { text: `⚠️ AI Analysis Failed. Raw Text:\n${ocrText}` };
  }
}

async function callLLMToChat(userText: string, ctx: Record<string, any>): Promise<{ text: string }> {
  const systemPrompt = `
You are a helpful assistant. Reply in a conversational, friendly tone.
Keep answers concise and ask a short follow-up question if it helps clarify the user's intent.
`;

  const start = Date.now();
  logInfo("llm.chat.request.start", { ...ctx, user_chars: userText.length });

  try {
    const res = await fetch(env.LLM_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.LLM_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        stream: false,
        model: env.LLM_MODEL,
        messages: [
          { role: "system", content: systemPrompt },
          { role: "user", content: userText },
        ],
        temperature: 0.7,
      }),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) {
      logError("llm.chat.request.failed", { ...ctx, ms: Date.now() - start, status: res.status, error: data?.error });
      throw new Error(data?.error?.message || `LLM API error (status ${res.status})`);
    }

    const content = data.choices?.[0]?.message?.content;

    logInfo("llm.chat.request.ok", { ...ctx, ms: Date.now() - start });

    return { text: content || "⚠️ I could not generate a reply." };
  } catch (error: any) {
    logError("llm.chat.request.exception", { ...ctx, ms: Date.now() - start, message: error?.message });
    return { text: "⚠️ AI reply failed. Please try again." };
  }
}

/* ----------------------------- Slack send utils -------------------------- */

async function postThreadOrChannel(
  client: any,
  args: { channel: string; thread_ts?: string; text: string },
  ctx: Record<string, any>,
) {
  try {
    return await client.chat.postMessage(args);
  } catch (err: any) {
    const slackErr = err?.data?.error || err?.message || String(err);
    logWarn("slack.chat.postMessage.failed", {
      ...ctx,
      slack_error: slackErr,
      channel: args.channel,
      thread_ts: args.thread_ts,
    });

    // Fallback if thread reply is not allowed
    if (slackErr === "cannot_reply_to_message" && args.thread_ts) {
      logWarn("slack.chat.postMessage.fallback_to_channel", { ...ctx, channel: args.channel });
      return await client.chat.postMessage({ channel: args.channel, text: args.text });
    }

    throw err;
  }
}

/* -------------------------------- Handlers ------------------------------- */

/**
 * 1) Slash command: /analyze
 * Slack requires ack within 3 seconds.
 */
app.command("/analyze", async ({ ack, respond, command, context }) => {
  const ctx = { event_id: context?.eventId, team_id: context?.teamId, kind: "slash:/analyze" };

  const t0 = Date.now();
  await ack();
  logInfo("slack.command.analyze.acked", { ...ctx, ack_ms: Date.now() - t0 });

  const text = (command.text || "").trim();
  if (!text) {
    await respond({ response_type: "ephemeral", text: "Send `/analyze <text>` or upload an image and mention me." });
    return;
  }

  await respond({ response_type: "ephemeral", text: "🤖 Analyzing your text..." });

  try {
    const { text: structuredResult } = await callLLMToChat(text, ctx);
    const MAX = 3500;
    const finalMsg = structuredResult.length > MAX ? `${structuredResult.slice(0, MAX)}...(truncated)` : structuredResult;

    await respond({ response_type: "ephemeral", text: finalMsg });
  } catch (err: any) {
    logError("slack.command.analyze.failed", { ...ctx, message: err?.message });
    await respond({ response_type: "ephemeral", text: `❌ Error: ${err instanceof Error ? err.message : String(err)}` });
  }
});

/**
 * 2) Channel message listener
 * Subscribe to message.channels and ensure bot is in the channel.
 */
app.event("message", async ({ event, client, context }) => {
  const e = event as any;

  const baseCtx = {
    event_id: context?.eventId,
    team_id: context?.teamId,
    kind: "event:message",
    channel: e.channel,
    user: e.user,
    ts: e.ts,
    subtype: e.subtype,
  };

  logInfo("slack.message.received", {
    ...baseCtx,
    has_files: Array.isArray(e.files) && e.files.length > 0,
    text_len: typeof e.text === "string" ? e.text.length : 0,
  });

  // ✅ Critical: only handle normal user messages (no subtype)
  if (e.subtype) {
    logInfo("slack.message.ignored_subtype", baseCtx);
    return;
  }
  if (!e.user) {
    logInfo("slack.message.ignored_no_user", baseCtx);
    return;
  }
  if (e.bot_id) {
    logInfo("slack.message.ignored_bot_id", baseCtx);
    return;
  }

  const threadTs = e.thread_ts ?? e.ts;

  // Image OCR pipeline
  const files = Array.isArray(e.files) ? e.files : [];
  const imageFile = files.find((f: any) => typeof f?.mimetype === "string" && f.mimetype.startsWith("image/"));

  if (imageFile?.url_private_download) {
    const ctx = { ...baseCtx, thread_ts: threadTs, file_id: imageFile.id, mimetype: imageFile.mimetype };

    await postThreadOrChannel(
      client,
      { channel: e.channel, thread_ts: threadTs, text: "🔍 Processing your image with OCR..." },
      ctx,
    );

    try {
      logInfo("pipeline.ocr.start", ctx);

      const imageBase64 = await fetchSlackFileAsBase64(imageFile.url_private_download, ctx);
      logInfo("pipeline.ocr.image_base64.ready", { ...ctx, b64_len: imageBase64.length });

      const ocrRawText = await callTencentOCRWithSDK(imageBase64, ctx);
      logInfo("pipeline.ocr.text.ready", { ...ctx, ocr_chars: ocrRawText.length });

      if (!ocrRawText) {
        await postThreadOrChannel(
          client,
          { channel: e.channel, thread_ts: threadTs, text: "❌ No text detected in the image." },
          ctx,
        );
        return;
      }

      const { text: structuredResult } = await callLLMToAnalyze(ocrRawText, ctx);
      const MAX = 3500;
      const finalMsg = structuredResult.length > MAX ? `${structuredResult.slice(0, MAX)}...(truncated)` : structuredResult;

      await postThreadOrChannel(client, { channel: e.channel, thread_ts: threadTs, text: finalMsg }, ctx);

      logInfo("pipeline.ocr.done", ctx);
    } catch (err: any) {
      logError("pipeline.ocr.failed", { ...ctx, message: err?.message, stack: err?.stack });
      await postThreadOrChannel(
        client,
        { channel: e.channel, thread_ts: threadTs, text: `❌ Error: ${err instanceof Error ? err.message : String(err)}` },
        ctx,
      );
    }
    return;
  }

  // Chat mode: if message mentions a user (<@...>), reply in thread
  if (typeof e.text === "string" && e.text.includes("<@")) {
    const cleanedText = e.text.replace(/<@[^>]+>/g, "").trim();
    const ctx = { ...baseCtx, thread_ts: threadTs, mode: "mention_chat" };

    if (!cleanedText) {
      await postThreadOrChannel(client, { channel: e.channel, thread_ts: threadTs, text: "Hi! Tell me what you need." }, ctx);
      return;
    }

    await postThreadOrChannel(client, { channel: e.channel, thread_ts: threadTs, text: "🤖 Thinking..." }, ctx);

    try {
      const { text: reply } = await callLLMToChat(cleanedText, ctx);
      const MAX = 3500;
      const finalMsg = reply.length > MAX ? `${reply.slice(0, MAX)}...(truncated)` : reply;
      await postThreadOrChannel(client, { channel: e.channel, thread_ts: threadTs, text: finalMsg }, ctx);
    } catch (err: any) {
      logError("pipeline.chat.failed", { ...ctx, message: err?.message, stack: err?.stack });
      await postThreadOrChannel(
        client,
        { channel: e.channel, thread_ts: threadTs, text: `❌ Error: ${err instanceof Error ? err.message : String(err)}` },
        ctx,
      );
    }
  }
});

/* --------------------------------- start -------------------------------- */

(async () => {
  logInfo("app.starting", { port: env.PORT, region: env.TENCENTCLOUD_REGION });

  await app.start(env.PORT);

  logInfo("app.started", {
    url: `http://localhost:${env.PORT}`,
    endpoints: {
      events: "/slack/events",
      commands: "/slack/commands",
      interactivity: "/slack/interactivity",
      health: "/health",
    },
  });

  console.log(`⚡ Slack bot listening on http://localhost:${env.PORT}`);
})();
