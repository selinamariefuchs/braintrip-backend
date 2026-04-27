import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import rateLimit from "express-rate-limit";
import helmet from "helmet";

dotenv.config();

if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is missing. Set it in .env or env vars.");
  process.exit(1);
}

if (!process.env.ANTHROPIC_API_KEY) {
  console.warn("⚠️  ANTHROPIC_API_KEY is missing. Anthropic proxy endpoints will not work.");
}

const app = express();

// Render runs behind a proxy — needed for accurate IP-based rate limiting
app.set("trust proxy", 1);

// Security headers (CSP, HSTS, X-Frame-Options, etc.)
app.use(helmet());

app.use(cors());
app.use(express.json({ limit: '20mb' }));

// ─── Rate limiters ──────────────────────────────────────────
// Strict limit on AI endpoints (each call costs real money)
const aiLimiter = rateLimit({
  windowMs: 60 * 1000,           // 1 minute
  max: 10,                        // 10 AI calls per IP per minute
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many requests. Please slow down and try again in a minute." },
});

// Looser global limit catches everything else
const globalLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 120,
  standardHeaders: true,
  legacyHeaders: false,
});

app.use(globalLimiter);

// Health endpoint — not rate-limited, used by Render uptime checks
app.get("/health", (req, res) => {
  res.json({ status: "ok", uptime: process.uptime() });
});

// ─── Anthropic safety guardrails ────────────────────────────
const ANTHROPIC_MAX_TOKENS_CAP = 4096;       // Hard cap regardless of what client sends
const ANTHROPIC_MAX_MESSAGES = 50;            // Prevent unbounded context bloating cost
const ANTHROPIC_REQUEST_TIMEOUT_MS = 60_000;  // Kill requests after 60s
const ANTHROPIC_ALLOWED_MODELS = new Set([
  "claude-sonnet-4-20250514",
  "claude-haiku-4-5-20251001",
  "claude-haiku-4-5",
]);

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.get("/", (req, res) => {
  res.send("BrainTrip backend is running 🚀");
});

// Landmark identification endpoint for Scan AI
app.post("/scan", aiLimiter, async (req, res) => {
  try {
    const { imageBase64, mimeType, city } = req.body ?? {};
    if (!imageBase64 || typeof imageBase64 !== "string") {
      return res.status(400).json({ error: "imageBase64 is required" });
    }

    const cityHint = city ? ` The user is in or near ${city}.` : "";

    const prompt = `You are BrainTrip's landmark scanner — an expert at identifying places from photos and telling their stories in a way that makes travelers say "wait, seriously?!"

Identify the SPECIFIC place in this image.${cityHint}

IDENTIFICATION RULES:
- Name the EXACT landmark, building, temple, monument, street, market, or site
- Use architecture, signage, landscape, skyline cues to narrow it down
- If 70%+ confident, commit to the specific name
- If it's a neighborhood, street, or market (not a single building), name that
- NEVER return vague labels like "regional architecture" or "historic building"
- If truly unidentifiable, set isLandmark to false and give your best guess

HOT FACTS RULES — this is the soul of the feature:
- Each fact should be 1 sentence, max 15 words
- Lead with the surprising part ("Built without a single nail" not "This temple was built without nails")
- Think: stories a local guide would whisper, not Wikipedia summaries
- Mix categories: one history/origin story, one weird/fun detail, one "most people don't know" secret

Return ONLY valid JSON, no markdown:
{
  "name": "Official name of the place",
  "location": "City, Country",
  "confidence": "high/medium/low",
  "isLandmark": true,
  "tagline": "One punchy line — why this place stops you in your tracks",
  "whyItMatters": "2 sentences max. The story behind the place, not a guidebook summary.",
  "hotFacts": [
    "Surprising origin or history fact (≤15 words)",
    "Weird, fun, or unexpected detail (≤15 words)",
    "Hidden secret most visitors walk right past (≤15 words)"
  ],
  "travelerTip": "One specific, practical tip for THIS exact spot"
}

If no landmark is visible return:
{ "isLandmark": false }`;

    const response = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "system",
          content: "You are an expert travel guide and landmark identification specialist.",
        },
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            { type: "image_url", image_url: { url: `data:${mimeType || "image/jpeg"};base64,${imageBase64}` } },
          ],
        },
      ],
      max_tokens: 800,
      temperature: 0.7,
    });

    let aiText = response.choices?.[0]?.message?.content || "";
    if (!aiText.trim()) {
      return res.status(500).json({ error: "No response from AI" });
    }
    // Remove markdown if present
    aiText = aiText.replace(/^```json\s*|```$/g, "").trim();
    let parsed;
    try {
      parsed = JSON.parse(aiText);
    } catch (error) {
      return res.status(500).json({ error: "Invalid JSON from AI", raw: aiText });
    }
    return res.json(parsed);
  } catch (error) {
    console.error("Scan AI error:", error);
    return res.status(500).json({ error: "Failed to analyze image" });
  }
});

app.post("/trivia", aiLimiter, async (req, res) => {
  try {
    const { city, mode, seenQuestions = [] } = req.body ?? {};

    if (!city || typeof city !== "string" || !city.trim()) {
      return res.status(400).json({ error: "City is required" });
    }

    const cleanCity = city.trim();
    const gameMode = mode === "challenge" ? "challenge" : "standard";

    const prompt = `
Generate 5 ${gameMode === "challenge" ? "VERY HARD" : "medium"} trivia questions about ${cleanCity}.

IMPORTANT:
- DO NOT repeat any of these questions:
${Array.isArray(seenQuestions) ? seenQuestions.join("\n") : ""}

- Questions must be unique, obscure, and specific
- Avoid common tourist facts
- Make them engaging and surprising

Return JSON format:
[
  {
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correctAnswer": "...",
    "funFact": "..."
  }
]
`;

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      input: [
        {
          role: "system",
          content:
            "You are a precise JSON generator for a travel trivia app. Keep answers short and clean.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 1.1,
      text: {
        format: {
          type: "json_schema",
          name: "braintrip_trivia",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: {
              questions: {
                type: "array",
                minItems: 5,
                maxItems: 5,
                items: {
                  type: "object",
                  additionalProperties: false,
                  properties: {
                    question: { type: "string" },
                    options: {
                      type: "array",
                      minItems: 4,
                      maxItems: 4,
                      items: { type: "string" },
                    },
                    correctAnswer: { type: "string" },
                    funFact: { type: "string" },
                    category: {
                      type: "string",
                      enum: [
                        "Neighborhoods",
                        "Food & Drink",
                        "Nightlife",
                        "Culture & Landmarks",
                        "Local Experience",
                      ],
                    },
                  },
                  required: [
                    "question",
                    "options",
                    "correctAnswer",
                    "funFact",
                    "category",
                  ],
                },
              },
            },
            required: ["questions"],
          },
        },
      },
    });

    console.log("OpenAI raw response:", JSON.stringify(response, null, 2));

    const aiText =
      response.output_text ||
      (response.output &&
        Array.isArray(response.output)
        ? response.output
            .map((item) => {
              if (!item?.content) return "";
              return item.content
                .map((c) => {
                  if (typeof c === "string") return c;
                  if (c?.type === "output_text") return c.text || "";
                  if (c?.type === "message" && typeof c.text === "string")
                    return c.text;
                  return "";
                })
                .join("");
            })
            .join("")
        : "") ||
      "";

    if (!aiText.trim()) {
      console.error("No text from OpenAI:", JSON.stringify(response, null, 2));
      return res.status(500).json({ error: "No response from AI" });
    }

    let parsed;
    try {
      parsed = JSON.parse(aiText);
    } catch (error) {
      console.error("Invalid JSON from AI:", aiText, error);
      return res.status(500).json({ error: "Invalid JSON from AI" });
    }

    if (!parsed?.questions || !Array.isArray(parsed.questions)) {
      console.error("Invalid structure from AI:", parsed);
      return res.status(500).json({ error: "Invalid trivia format from AI" });
    }

    const cleanedQuestions = parsed.questions
      .map((q) => {
        const options = Array.isArray(q.options) ? q.options : [];
        const correctAnswer =
          typeof q.correctAnswer === "string" && options.includes(q.correctAnswer)
            ? q.correctAnswer
            : options[0] || "";

        return {
          question: typeof q.question === "string" ? q.question : "",
          options,
          correctAnswer,
          funFact: typeof q.funFact === "string" ? q.funFact : "",
          category:
            typeof q.category === "string" ? q.category : "Local Experience",
        };
      })
      .filter(
        (q) =>
          q.question &&
          q.options.length === 4 &&
          q.correctAnswer &&
          q.funFact)
      .slice(0, 5);

    const uniqueQuestions = cleanedQuestions.filter(
      (q) => !seenQuestions.includes(q.question)
    );
    const finalQuestions =
      uniqueQuestions.length >= 5 ? uniqueQuestions : cleanedQuestions;

    return res.json({
      city: cleanCity,
      mode: gameMode,
      questions: finalQuestions,
    });
  } catch (error) {
    console.error("Trivia generation error:", error);
    return res.status(500).json({ error: "Failed to generate trivia" });
  }
});

app.post("/itinerary", aiLimiter, async (req, res) => {
  try {
    const { city, category = "all" } = req.body ?? {};

    if (!city || typeof city !== "string" || !city.trim()) {
      return res.status(400).json({ error: "City is required" });
    }

    const cleanCity = city.trim();

    const categoryPrompt = category === "attractions"
      ? "Focus on iconic landmarks and attractions only."
      : category === "food"
      ? "Focus on restaurants, cafes, and food experiences only."
      : category === "things-to-do"
      ? "Focus on activities, experiences, and things to do only."
      : "Mix iconic landmarks, local food spots, and hidden gems equally.";

    const prompt = `
Generate 5 must-visit spots for ${cleanCity}.
${categoryPrompt}

Requirements:
- Each spot must be a real specific place
- Include a compelling reason why travelers love it
- Mix well known spots with hidden gems
- Keep descriptions fun and engaging
- Never include fake or generic places

Return exactly 5 spots.
`;

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      input: [
        {
          role: "system",
          content: "You are a travel expert generating must-visit spot recommendations for a travel app. Keep all responses specific, accurate, and engaging.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 1.0,
      text: {
        format: {
          type: "json_schema",
          name: "braintrip_itinerary",
          strict: true,
          schema: {
            type: "object",
            additionalProperties: false,
            properties: {
              spots: {
                type: "array",
                minItems: 5,
                maxItems: 5,
                items: {
                  type: "object",
                  additionalProperties: false,
                  properties: {
                    name: { type: "string" },
                    type: {
                      type: "string",
                      enum: [
                        "Landmark",
                        "Food & Drink",
                        "Hidden Gem",
                        "Experience",
                        "Culture",
                      ],
                    },
                    description: { type: "string" },
                    whyGo: { type: "string" },
                  },
                  required: ["name", "type", "description", "whyGo"],
                },
              },
            },
            required: ["spots"],
          },
        },
      },
    });

    const aiText = response.output_text ||
      (response.output && Array.isArray(response.output)
        ? response.output.map((item) => {
            if (!item?.content) return "";
            return item.content.map((c) => {
              if (typeof c === "string") return c;
              if (c?.type === "output_text") return c.text || "";
              return "";
            }).join("");
          }).join("")
        : "") || "";

    if (!aiText.trim()) {
      return res.status(500).json({ error: "No response from AI" });
    }

    let parsed;
    try {
      parsed = JSON.parse(aiText);
    } catch (error) {
      return res.status(500).json({ error: "Invalid JSON from AI" });
    }

    if (!parsed?.spots || !Array.isArray(parsed.spots)) {
      return res.status(500).json({ error: "Invalid itinerary format from AI" });
    }

    const cleanedSpots = parsed.spots
      .map((s) => ({
        name: typeof s.name === "string" ? s.name : "",
        type: typeof s.type === "string" ? s.type : "Highlight",
        description: typeof s.description === "string" ? s.description : "",
        whyGo: typeof s.whyGo === "string" ? s.whyGo : "",
      }))
      .filter((s) => s.name && s.description && s.whyGo)
      .slice(0, 5);

    return res.json({
      city: cleanCity,
      category,
      spots: cleanedSpots,
    });

  } catch (error) {
    console.error("Itinerary generation error:", error);
    return res.status(500).json({ error: "Failed to generate itinerary" });
  }
});

// Generic Anthropic Messages API proxy
// Used by trivia-quick, use-scan-insight-api, use-city-validation, etc.
app.post("/anthropic/messages", aiLimiter, async (req, res) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), ANTHROPIC_REQUEST_TIMEOUT_MS);

  try {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: "Service temporarily unavailable" });
    }

    const { model, max_tokens, messages } = req.body ?? {};

    if (!model || !max_tokens || !messages) {
      return res.status(400).json({ error: "model, max_tokens, and messages are required" });
    }

    // Guardrail 1: Reject unapproved models (don't let clients call expensive Opus)
    if (!ANTHROPIC_ALLOWED_MODELS.has(model)) {
      return res.status(400).json({ error: "Model not allowed" });
    }

    // Guardrail 2: Cap max_tokens server-side regardless of client value
    const safeMaxTokens = Math.min(Math.max(1, Number(max_tokens) || 0), ANTHROPIC_MAX_TOKENS_CAP);

    // Guardrail 3: Cap message count to prevent context bloat
    if (!Array.isArray(messages) || messages.length === 0 || messages.length > ANTHROPIC_MAX_MESSAGES) {
      return res.status(400).json({ error: "Invalid messages array" });
    }

    const response = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({ model, max_tokens: safeMaxTokens, messages }),
      signal: controller.signal,
    });

    const data = await response.json();

    if (!response.ok) {
      // Don't leak Anthropic's internal error details to clients
      console.error("Anthropic upstream error:", response.status, data?.error?.message);
      return res.status(response.status >= 500 ? 502 : response.status).json({
        error: response.status === 429 ? "Rate limited upstream, try again shortly" : "Upstream service error",
      });
    }

    return res.json(data);
  } catch (error) {
    if (error?.name === "AbortError") {
      console.warn("Anthropic request timed out");
      return res.status(504).json({ error: "Request timed out" });
    }
    console.error("Anthropic proxy error:", error?.message || error);
    return res.status(500).json({ error: "Failed to proxy request" });
  } finally {
    clearTimeout(timeoutId);
  }
});

const PORT = process.env.PORT || 10000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
