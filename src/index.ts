import "dotenv/config";
import SlackBolt from "@slack/bolt";
import * as tencentcloud from "tencentcloud-sdk-nodejs-ocr";
import { z } from "zod";

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

// 用 ExpressReceiver 才能自定义 path (/slack/events /slack/commands /slack/interactivity)
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

const app = new App({
  token: env.SLACK_BOT_TOKEN,
  receiver,
});

const OcrClient = tencentcloud.ocr.v20181119.Client;
const ocrClient = new OcrClient({
  credential: {
    secretId: env.TENCENTCLOUD_SECRET_ID,
    secretKey: env.TENCENTCLOUD_SECRET_KEY,
  },
  region: env.TENCENTCLOUD_REGION,
  profile: { httpProfile: { endpoint: "ocr.tencentcloudapi.com" } },
});

async function fetchSlackFileAsBase64(fileUrl: string): Promise<string> {
  const res = await fetch(fileUrl, {
    headers: {
      Authorization: `Bearer ${env.SLACK_BOT_TOKEN}`,
    },
  });
  if (!res.ok) {
    throw new Error(`Failed to download file: ${res.status}`);
  }
  const arrayBuffer = await res.arrayBuffer();
  return Buffer.from(arrayBuffer).toString("base64");
}

async function callTencentOCRWithSDK(imageBase64: string): Promise<string> {
  try {
    const result = await ocrClient.GeneralAccurateOCR({ ImageBase64: imageBase64 });
    const dets = (result.TextDetections || []) as Array<{ DetectedText?: string }>;
    if (dets.length === 0) return "";
    return dets
      .map((x) => x.DetectedText)
      .filter((x) => x && x.trim().length > 0)
      .join("\n");
  } catch (err) {
    console.error("Tencent SDK Error:", err);
    throw new Error(`OCR Failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

async function callLLMToAnalyze(
  ocrText: string,
): Promise<{ text: string; usage?: { prompt_tokens: number; completion_tokens: number } }> {
  const today = new Date().toISOString().split("T")[0];
  const systemPrompt = `
You are an expert OCR Data Extraction Auditor. Your goal is to extract precise structured data from receipts/invoices.

**Current Server Date:** ${today} (YYYY-MM-DD)
*Use this date to infer the year if missing, or to validate that the transaction date is not in the distant future.*

**Critical Instruction: Chain-of-Thought Analysis**
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

    const data = await res.json();
    if (data.error) {
      console.error("LLM API Error Details:", data.error);
      throw new Error(data.error.message || "LLM API returned an error");
    }

    const content = data.choices?.[0]?.message?.content;
    const usage = data.usage;

    return {
      text: content || "⚠️ AI could not analyze the text.",
      usage,
    };
  } catch (error) {
    console.error("Call LLM Failed:", error);
    return { text: `⚠️ AI Analysis Failed. Raw Text:\n${ocrText}` };
  }
}

async function callLLMToChat(userText: string): Promise<{ text: string }> {
  const systemPrompt = `
You are a helpful assistant. Reply in a conversational, friendly tone.
Keep answers concise and ask a short follow-up question if it helps clarify the user's intent.
`;

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

    const data = await res.json();
    if (data.error) {
      console.error("LLM API Error Details:", data.error);
      throw new Error(data.error.message || "LLM API returned an error");
    }

    const content = data.choices?.[0]?.message?.content;
    return { text: content || "⚠️ I could not generate a reply." };
  } catch (error) {
    console.error("Call LLM Failed:", error);
    return { text: "⚠️ AI reply failed. Please try again." };
  }
}

/**
 * 1) Slash command: /analyze
 * Slack 要求 3 秒内 ack，否则会重试。Bolt 的 ack() 就是为这个设计的。
 */
app.command("/analyze", async ({ ack, respond, command }) => {
  await ack();
  const text = (command.text || "").trim();

  if (!text) {
    await respond({
      response_type: "ephemeral",
      text: "Send `/analyze <text>` or upload an image and mention me.",
    });
    return;
  }

  await respond({
    response_type: "ephemeral",
    text: "🤖 Analyzing your text...",
  });

  try {
    const { text: structuredResult } = await callLLMToChat(text);
    const MAX = 3500;
    const finalMsg =
      structuredResult.length > MAX ? `${structuredResult.slice(0, MAX)}...(truncated)` : structuredResult;

    await respond({
      response_type: "ephemeral",
      text: finalMsg,
    });
  } catch (err) {
    await respond({
      response_type: "ephemeral",
      text: `❌ Error: ${err instanceof Error ? err.message : String(err)}`,
    });
  }
});

/**
 * 2) 监听频道消息：只做最小例子
 * 注意：要在 Slack 后台订阅 message.channels，并且 bot 要在频道里
 */
app.event("message", async ({ event, client }) => {
  // @ts-ignore
  const e = event as any;

  // 忽略 bot 自己消息，避免自触发
  if (e.subtype === "bot_message") return;

  const files = Array.isArray(e.files) ? e.files : [];
  const imageFile = files.find(
    (f: any) => typeof f?.mimetype === "string" && f.mimetype.startsWith("image/"),
  );

  if (imageFile?.url_private_download) {
    await client.chat.postMessage({
      channel: e.channel,
      thread_ts: e.ts,
      text: "🔍 Processing your image with OCR...",
    });

    try {
      const imageBase64 = await fetchSlackFileAsBase64(imageFile.url_private_download);
      const ocrRawText = await callTencentOCRWithSDK(imageBase64);

      if (!ocrRawText) {
        await client.chat.postMessage({
          channel: e.channel,
          thread_ts: e.ts,
          text: "❌ No text detected in the image.",
        });
        return;
      }

      const { text: structuredResult } = await callLLMToAnalyze(ocrRawText);
      const MAX = 3500;
      const finalMsg =
        structuredResult.length > MAX ? `${structuredResult.slice(0, MAX)}...(truncated)` : structuredResult;

      await client.chat.postMessage({
        channel: e.channel,
        thread_ts: e.ts,
        text: finalMsg,
      });
    } catch (err) {
      console.error("OCR/LLM Error:", err);
      await client.chat.postMessage({
        channel: e.channel,
        thread_ts: e.ts,
        text: `❌ Error: ${err instanceof Error ? err.message : String(err)}`,
      });
    }

    return;
  }

  // 如果用户 @bot，用对话式 AI 回复
  if (typeof e.text === "string" && e.text.includes("<@")) {
    const cleanedText = e.text.replace(/<@[^>]+>/g, "").trim();

    if (!cleanedText) {
      await client.chat.postMessage({
        channel: e.channel,
        thread_ts: e.ts,
        text: "Hi! Tell me what you need help with.",
      });
      return;
    }

    await client.chat.postMessage({
      channel: e.channel,
      thread_ts: e.ts,
      text: "🤖 Thinking...",
    });

    try {
      const { text: reply } = await callLLMToChat(cleanedText);
      const MAX = 3500;
      const finalMsg = reply.length > MAX ? `${reply.slice(0, MAX)}...(truncated)` : reply;

      await client.chat.postMessage({
        channel: e.channel,
        thread_ts: e.ts,
        text: finalMsg,
      });
    } catch (err) {
      await client.chat.postMessage({
        channel: e.channel,
        thread_ts: e.ts,
        text: `❌ Error: ${err instanceof Error ? err.message : String(err)}`,
      });
    }
  }
});

(async () => {
  await app.start(env.PORT);
  console.log(`⚡ Slack bot listening on http://localhost:${env.PORT}`);
})();
