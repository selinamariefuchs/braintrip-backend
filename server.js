import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";
import rateLimit from "express-rate-limit";
import helmet from "helmet";
import crypto from "crypto";
import { createClient } from "@supabase/supabase-js";

dotenv.config();

if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is missing. Set it in .env or env vars.");
  process.exit(1);
}

if (!process.env.ANTHROPIC_API_KEY) {
  console.warn("⚠️  ANTHROPIC_API_KEY is missing. Anthropic proxy endpoints will not work.");
}

// Drive Mode TTS uses OpenAI (key already required above)

// Supabase admin client for caching TTS audio in Storage
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const supabase = (supabaseUrl && supabaseServiceKey)
  ? createClient(supabaseUrl, supabaseServiceKey, { auth: { persistSession: false } })
  : null;

if (!supabase) {
  console.warn("⚠️  Supabase service role missing. TTS audio will not be cached (each request will hit ElevenLabs).");
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

// ─── Auth middleware ────────────────────────────────────────
// Validates the Supabase JWT in the Authorization header so AI endpoints
// can only be called by signed-in BrainTrip users. Without this gate the
// /anthropic/messages proxy is wide open — anyone with the URL could burn
// the project's Anthropic credits. Guests skip this gate via a guest token
// (signed short-lived nonce) issued at app launch — see /guest-token below.
const GUEST_TOKEN_SECRET = process.env.GUEST_TOKEN_SECRET || "";
const GUEST_TOKEN_TTL_MS = 60 * 60 * 1000; // 1 hour

// Graceful-degradation flag. If GUEST_TOKEN_SECRET is not configured, the
// guest-token issue/verify path can't work — without this fallback the app
// would hard-fail every AI call for non-signed-in users. We log a loud
// warning at startup so it's not forgotten, but allow unauthenticated
// requests through. Once the secret is set in Render, this branch is
// silently bypassed and full auth gating kicks in.
const AUTH_OPEN_MODE = !GUEST_TOKEN_SECRET;
if (AUTH_OPEN_MODE) {
  console.warn(
    "⚠️  GUEST_TOKEN_SECRET not set — running in OPEN AUTH MODE.\n" +
    "    AI endpoints accept unauthenticated requests so the app keeps\n" +
    "    working. Set GUEST_TOKEN_SECRET in your Render env vars to\n" +
    "    enable proper auth gating before public launch."
  );
}

function verifyGuestToken(token) {
  if (!GUEST_TOKEN_SECRET || typeof token !== "string") return false;
  const parts = token.split(".");
  if (parts.length !== 2) return false;
  const [payloadB64, sig] = parts;
  const expected = crypto.createHmac("sha256", GUEST_TOKEN_SECRET).update(payloadB64).digest("hex");
  if (!crypto.timingSafeEqual(Buffer.from(sig, "hex"), Buffer.from(expected, "hex"))) return false;
  try {
    const payload = JSON.parse(Buffer.from(payloadB64, "base64url").toString("utf8"));
    if (typeof payload.exp !== "number") return false;
    return Date.now() < payload.exp;
  } catch {
    return false;
  }
}

async function requireAuth(req, res, next) {
  // Open auth mode (env var unset). Allow through, mark as guest. Log only
  // a debug line to avoid spamming.
  if (AUTH_OPEN_MODE) {
    req.isGuest = true;
    req.authOpenMode = true;
    return next();
  }

  const header = req.headers.authorization || "";
  const match = header.match(/^Bearer\s+(.+)$/i);
  if (!match) {
    return res.status(401).json({ error: "Missing Authorization header" });
  }
  const token = match[1].trim();

  // 1) Try Supabase JWT (signed-in users)
  if (supabase) {
    try {
      const { data, error } = await supabase.auth.getUser(token);
      if (!error && data?.user?.id) {
        req.userId = data.user.id;
        req.isGuest = false;
        return next();
      }
    } catch {
      // fall through to guest check
    }
  }

  // 2) Try guest token (signed short-lived nonce)
  if (verifyGuestToken(token)) {
    req.isGuest = true;
    return next();
  }

  return res.status(401).json({ error: "Invalid or expired token" });
}

// Issue a guest token — used by the app's guest-mode users (those who haven't
// signed up). 1-hour TTL, HMAC-signed so it can't be forged. The client
// transparently refreshes when it expires. Rate-limited so this isn't itself
// abusable.
app.post("/guest-token", globalLimiter, (req, res) => {
  if (!GUEST_TOKEN_SECRET) {
    return res.status(503).json({ error: "Guest mode not configured" });
  }
  const payload = { exp: Date.now() + GUEST_TOKEN_TTL_MS, kind: "guest" };
  const payloadB64 = Buffer.from(JSON.stringify(payload), "utf8").toString("base64url");
  const sig = crypto.createHmac("sha256", GUEST_TOKEN_SECRET).update(payloadB64).digest("hex");
  res.json({ token: `${payloadB64}.${sig}`, expiresAt: payload.exp });
});

// Health endpoint — not rate-limited or auth-gated, used by Render uptime checks
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

app.get("/app-ads.txt", (req, res) => {
  res.type("text/plain").send("google.com, pub-4241252919358872, DIRECT, f08c47fec0942fa0\n");
});

app.get("/", (req, res) => {
  res.send("BrainTrip backend is running 🚀");
});

// ─── Spot photo proxy ───────────────────────────────────────
// Resolves "spot name, city" → one Google Places photo, redirecting the
// client to the key-free googleusercontent URL. The API key never leaves
// the server. Results (including misses) are cached in memory since spot
// photos are effectively static.
const SPOT_PHOTO_CACHE = new Map(); // queryLower -> { url: string | null }

app.get("/spot-photo", globalLimiter, async (req, res) => {
  const key = process.env.GOOGLE_PLACES_API_KEY;
  if (!key) return res.status(503).json({ error: "Places not configured" });

  const query = String(req.query.q || "").trim().slice(0, 200);
  if (!query) return res.status(400).json({ error: "q is required" });

  const cacheKey = query.toLowerCase();
  const hit = SPOT_PHOTO_CACHE.get(cacheKey);
  if (hit) {
    if (!hit.url) return res.status(404).json({ error: "No photo found" });
    return res.redirect(302, hit.url);
  }

  try {
    const findRes = await fetch(
      `https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input=${encodeURIComponent(query)}&inputtype=textquery&fields=photos&key=${key}`
    );
    const findData = await findRes.json();
    const ref = findData?.candidates?.[0]?.photos?.[0]?.photo_reference;
    if (!ref) {
      SPOT_PHOTO_CACHE.set(cacheKey, { url: null });
      return res.status(404).json({ error: "No photo found" });
    }

    // The photo endpoint 302s to a googleusercontent URL that doesn't
    // contain the API key — resolve it and send the client there.
    const photoRes = await fetch(
      `https://maps.googleapis.com/maps/api/place/photo?maxwidth=640&photo_reference=${encodeURIComponent(ref)}&key=${key}`,
      { redirect: "manual" }
    );
    const loc = photoRes.headers.get("location");
    if (!loc) {
      SPOT_PHOTO_CACHE.set(cacheKey, { url: null });
      return res.status(404).json({ error: "No photo found" });
    }

    SPOT_PHOTO_CACHE.set(cacheKey, { url: loc });
    return res.redirect(302, loc);
  } catch (err) {
    console.warn("[spot-photo] lookup failed:", err?.message || err);
    return res.status(500).json({ error: "Photo lookup failed" });
  }
});

// ─── Place search (custom spot picker) ──────────────────────
// Lets users add a place they already know: resolves free text to up to 5
// Places candidates with name, address, rating, and a photo served via
// /photo-ref. Cached per query.
const PLACE_SEARCH_CACHE = new Map(); // queryLower -> results array

app.get("/place-search", globalLimiter, async (req, res) => {
  const key = process.env.GOOGLE_PLACES_API_KEY;
  if (!key) return res.status(503).json({ error: "Places not configured" });
  const query = String(req.query.q || "").trim().slice(0, 200);
  if (!query) return res.status(400).json({ error: "q is required" });

  const cacheKey = query.toLowerCase();
  const hit = PLACE_SEARCH_CACHE.get(cacheKey);
  if (hit) return res.json({ results: hit });

  try {
    const searchRes = await fetch(
      `https://maps.googleapis.com/maps/api/place/textsearch/json?query=${encodeURIComponent(query)}&key=${key}`
    );
    const data = await searchRes.json();
    const results = (Array.isArray(data?.results) ? data.results : [])
      .slice(0, 5)
      .map((r) => ({
        name: r.name,
        address: r.formatted_address || "",
        rating: typeof r.rating === "number" ? r.rating : null,
        totalRatings: typeof r.user_ratings_total === "number" ? r.user_ratings_total : null,
        photoRef: r.photos?.[0]?.photo_reference || null,
      }));
    PLACE_SEARCH_CACHE.set(cacheKey, results);
    return res.json({ results });
  } catch (err) {
    console.warn("[place-search] failed:", err?.message || err);
    return res.status(500).json({ error: "Search failed" });
  }
});

// Serves a photo by Places photo_reference (from /place-search results).
const PHOTO_REF_CACHE = new Map(); // ref -> googleusercontent url

app.get("/photo-ref", globalLimiter, async (req, res) => {
  const key = process.env.GOOGLE_PLACES_API_KEY;
  if (!key) return res.status(503).json({ error: "Places not configured" });
  const ref = String(req.query.ref || "").trim().slice(0, 600);
  if (!ref) return res.status(400).json({ error: "ref is required" });

  const cached = PHOTO_REF_CACHE.get(ref);
  if (cached) return res.redirect(302, cached);
  try {
    const photoRes = await fetch(
      `https://maps.googleapis.com/maps/api/place/photo?maxwidth=640&photo_reference=${encodeURIComponent(ref)}&key=${key}`,
      { redirect: "manual" }
    );
    const loc = photoRes.headers.get("location");
    if (!loc) return res.status(404).json({ error: "No photo" });
    PHOTO_REF_CACHE.set(ref, loc);
    return res.redirect(302, loc);
  } catch (err) {
    return res.status(500).json({ error: "Photo lookup failed" });
  }
});

// ─── Nearby landmark (daily photo challenge) ────────────────
// Finds the most prominent tourist attraction within ~2km of the given
// coordinates. Cached per ~1km grid cell — landmarks don't move.
const NEARBY_CACHE = new Map(); // cellKey -> landmark | null

app.get("/nearby-landmark", globalLimiter, async (req, res) => {
  const key = process.env.GOOGLE_PLACES_API_KEY;
  if (!key) return res.status(503).json({ error: "Places not configured" });

  const lat = Number(req.query.lat);
  const lng = Number(req.query.lng);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    return res.status(400).json({ error: "lat and lng are required" });
  }

  const cellKey = `${lat.toFixed(2)},${lng.toFixed(2)}`;
  if (NEARBY_CACHE.has(cellKey)) {
    const hit = NEARBY_CACHE.get(cellKey);
    return hit ? res.json(hit) : res.status(404).json({ error: "No landmark nearby" });
  }

  try {
    const searchRes = await fetch(
      `https://maps.googleapis.com/maps/api/place/nearbysearch/json?location=${lat},${lng}&radius=2000&type=tourist_attraction&key=${key}`
    );
    const data = await searchRes.json();
    const results = Array.isArray(data?.results) ? data.results : [];
    // Most-reviewed = most recognizable target for the challenge.
    const best = results
      .filter((r) => (r.user_ratings_total || 0) >= 50)
      .sort((a, b) => (b.user_ratings_total || 0) - (a.user_ratings_total || 0))[0];
    if (!best) {
      NEARBY_CACHE.set(cellKey, null);
      return res.status(404).json({ error: "No landmark nearby" });
    }
    const payload = {
      name: best.name,
      vicinity: best.vicinity || "",
      rating: typeof best.rating === "number" ? best.rating : null,
      totalRatings: best.user_ratings_total || null,
      photoRef: best.photos?.[0]?.photo_reference || null,
    };
    NEARBY_CACHE.set(cellKey, payload);
    return res.json(payload);
  } catch (err) {
    console.warn("[nearby-landmark] failed:", err?.message || err);
    return res.status(500).json({ error: "Lookup failed" });
  }
});

// ─── Spot info: rating + condensed real reviews ─────────────
// Returns the Google rating and short excerpts from real reviews for a
// spot — the "why visitors love it" expansion in the app. Cached.
const SPOT_INFO_CACHE = new Map(); // queryLower -> payload

app.get("/spot-info", globalLimiter, async (req, res) => {
  const key = process.env.GOOGLE_PLACES_API_KEY;
  if (!key) return res.status(503).json({ error: "Places not configured" });
  const query = String(req.query.q || "").trim().slice(0, 200);
  if (!query) return res.status(400).json({ error: "q is required" });

  const cacheKey = query.toLowerCase();
  const hit = SPOT_INFO_CACHE.get(cacheKey);
  if (hit) return res.json(hit);

  try {
    const findRes = await fetch(
      `https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input=${encodeURIComponent(query)}&inputtype=textquery&fields=place_id&key=${key}`
    );
    const findData = await findRes.json();
    const placeId = findData?.candidates?.[0]?.place_id;
    if (!placeId) return res.status(404).json({ error: "Place not found" });

    const detailsRes = await fetch(
      `https://maps.googleapis.com/maps/api/place/details/json?place_id=${encodeURIComponent(placeId)}&fields=rating,user_ratings_total,reviews&reviews_sort=most_relevant&key=${key}`
    );
    const detailsData = await detailsRes.json();
    const result = detailsData?.result || {};
    const reviews = (Array.isArray(result.reviews) ? result.reviews : [])
      .filter((r) => typeof r.text === "string" && r.text.length > 30 && r.rating >= 4)
      .slice(0, 3)
      .map((r) => ({
        rating: r.rating,
        // First ~180 chars, cut at a sentence/word boundary
        excerpt: r.text.slice(0, 180).replace(/\s+\S*$/, "") + (r.text.length > 180 ? "…" : ""),
      }));

    const payload = {
      rating: typeof result.rating === "number" ? result.rating : null,
      totalRatings: typeof result.user_ratings_total === "number" ? result.user_ratings_total : null,
      reviews,
    };
    SPOT_INFO_CACHE.set(cacheKey, payload);
    return res.json(payload);
  } catch (err) {
    console.warn("[spot-info] failed:", err?.message || err);
    return res.status(500).json({ error: "Info lookup failed" });
  }
});

// Landmark identification endpoint for Scan AI
app.post("/scan", aiLimiter, requireAuth, async (req, res) => {
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

app.post("/trivia", aiLimiter, requireAuth, async (req, res) => {
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
    console.error("Trivia generation error:", error?.stack || error);
    return res.status(500).json({
      error: "Failed to generate trivia",
      reason: String(error?.message || error).slice(0, 300),
      code: error?.status || error?.code || null,
    });
  }
});

app.post("/itinerary", aiLimiter, requireAuth, async (req, res) => {
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
- For whyGo, write a condensed summary of what real visitors consistently praise — the specific things reviewers actually mention (a dish, a view, a time of day, a detail). Written like a distilled Google-reviews overview, e.g. "Visitors rave about the custard tarts straight from the oven and say the line moves faster than it looks" — NEVER generic praise like "a must-see with something for everyone"
- Mix well known spots with hidden gems
- Keep descriptions fun and engaging
- Never include fake or generic places
- For each spot, include the neighborhood or district name (e.g. "Asakusa", "Shibuya", "Le Marais")
- For each spot, rate the price level: 1 = free, 2 = budget ($1-15), 3 = moderate ($15-50), 4 = expensive ($50+)
- For each spot, indicate if advance tickets or reservations are typically needed (true/false)

For each spot, write a photoPrompt — a creative, specific photo challenge that makes the traveler engage with the place. These should feel like a photographer friend whispering "you HAVE to get this shot." Examples of great prompts:
- "Capture the view from the observation deck with the river curving below"
- "Find the mosaic floor in the main hall and snap it before the crowd arrives"
- "Get the street food sizzling on the grill with steam rising"
Bad prompts (too generic): "Take a photo here", "Snap a pic at this place"

Also generate 3 bonus hidden challenges — city-wide photo missions not tied to a specific spot. These should be distinctive to ${cleanCity} (local culture, architecture, street life, food scene). Examples:
- "Find a bodega cat in Manhattan"
- "Capture the golden light hitting the Seine at dusk"
- "Photograph the oldest piece of street art you can find in Shoreditch"

Return exactly 5 spots and 3 hidden challenges.
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
                    photoPrompt: { type: "string" },
                    neighborhood: { type: "string" },
                    priceLevel: { type: "number", enum: [1, 2, 3, 4] },
                    ticketsNeeded: { type: "boolean" },
                  },
                  required: ["name", "type", "description", "whyGo", "photoPrompt", "neighborhood", "priceLevel", "ticketsNeeded"],
                },
              },
              hiddenChallenges: {
                type: "array",
                minItems: 3,
                maxItems: 3,
                items: {
                  type: "object",
                  additionalProperties: false,
                  properties: {
                    prompt: { type: "string" },
                  },
                  required: ["prompt"],
                },
              },
            },
            required: ["spots", "hiddenChallenges"],
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
        photoPrompt: typeof s.photoPrompt === "string" ? s.photoPrompt : "",
        neighborhood: typeof s.neighborhood === "string" ? s.neighborhood : "",
        priceLevel: [1, 2, 3, 4].includes(s.priceLevel) ? s.priceLevel : 1,
        ticketsNeeded: typeof s.ticketsNeeded === "boolean" ? s.ticketsNeeded : false,
      }))
      .filter((s) => s.name && s.description && s.whyGo)
      .slice(0, 5);

    const cleanedHidden = (parsed.hiddenChallenges || [])
      .map((h) => ({ prompt: typeof h.prompt === "string" ? h.prompt : "" }))
      .filter((h) => h.prompt)
      .slice(0, 3);

    return res.json({
      city: cleanCity,
      category,
      spots: cleanedSpots,
      hiddenChallenges: cleanedHidden,
    });

  } catch (error) {
    console.error("Itinerary generation error:", error?.stack || error);
    return res.status(500).json({
      error: "Failed to generate itinerary",
      reason: String(error?.message || error).slice(0, 300),
      code: error?.status || error?.code || null,
    });
  }
});

// Generic Anthropic Messages API proxy
// Used by trivia-quick, use-scan-insight-api, use-city-validation, etc.
app.post("/anthropic/messages", aiLimiter, requireAuth, async (req, res) => {
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

// ─── OpenAI TTS proxy with caching ───────────────────────
// Used by Drive Mode to read trivia questions / answers / fun facts aloud.
// Audio is cached in Supabase Storage by content hash to minimize costs.

const TTS_BUCKET = "tts-cache"; // Must be created as a public bucket in Supabase Storage
const OPENAI_DEFAULT_VOICE = "nova"; // warm, conversational
const OPENAI_TTS_MODEL = "tts-1-hd"; // higher quality
const OPENAI_TTS_VOICES = new Set(["alloy", "echo", "fable", "onyx", "nova", "shimmer"]);
const TTS_MAX_TEXT_LENGTH = 2000; // Server-side cap

const ttsLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30, // Plenty for a quiz session (7 questions × ~3 audio clips each)
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: "Too many TTS requests. Slow down." },
});

function ttsCacheKey(text, voice) {
  const hash = crypto.createHash("sha256").update(`${OPENAI_TTS_MODEL}:${voice}:${text}`).digest("hex");
  return `${hash}.mp3`;
}

async function getCachedAudio(text, voiceId) {
  if (!supabase) return null;
  const fileName = ttsCacheKey(text, voiceId);
  try {
    const { data } = supabase.storage.from(TTS_BUCKET).getPublicUrl(fileName);
    if (!data?.publicUrl) return null;
    // Check if the file actually exists by attempting a HEAD request
    const res = await fetch(data.publicUrl, { method: "HEAD" });
    if (res.ok) return data.publicUrl;
    return null;
  } catch {
    return null;
  }
}

async function cacheAudio(text, voiceId, audioBuffer) {
  if (!supabase) return null;
  const fileName = ttsCacheKey(text, voiceId);
  try {
    const { error } = await supabase.storage
      .from(TTS_BUCKET)
      .upload(fileName, audioBuffer, {
        contentType: "audio/mpeg",
        upsert: false,
      });
    if (error && !error.message.includes("already exists")) {
      console.warn("[tts] cache upload error:", error.message);
      return null;
    }
    const { data } = supabase.storage.from(TTS_BUCKET).getPublicUrl(fileName);
    return data?.publicUrl ?? null;
  } catch (err) {
    console.warn("[tts] cache upload threw:", err?.message || err);
    return null;
  }
}

app.post("/tts", aiLimiter, ttsLimiter, requireAuth, async (req, res) => {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return res.status(503).json({ error: "TTS unavailable" });
  }

  const { text, voice: voiceParam } = req.body ?? {};
  if (!text || typeof text !== "string" || !text.trim()) {
    return res.status(400).json({ error: "text is required" });
  }

  const safeText = text.slice(0, TTS_MAX_TEXT_LENGTH);
  const requestedVoice = (typeof voiceParam === "string" ? voiceParam.trim().toLowerCase() : "");
  const voice = OPENAI_TTS_VOICES.has(requestedVoice) ? requestedVoice : OPENAI_DEFAULT_VOICE;

  // 1. Try cache first
  try {
    const cachedUrl = await getCachedAudio(safeText, voice);
    if (cachedUrl) {
      return res.json({ url: cachedUrl, cached: true });
    }
  } catch {}

  // 2. Generate via OpenAI TTS
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 30_000);

  try {
    const response = await fetch(
      "https://api.openai.com/v1/audio/speech",
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          "Accept": "audio/mpeg",
        },
        body: JSON.stringify({
          model: OPENAI_TTS_MODEL,
          voice,
          input: safeText,
          response_format: "mp3",
          speed: 1.0,
        }),
        signal: controller.signal,
      }
    );

    if (!response.ok) {
      const errText = await response.text().catch(() => "");
      console.error("[tts] OpenAI error:", response.status, "voice=", voice, "body=", errText.slice(0, 400));
      return res.status(response.status >= 500 ? 502 : response.status).json({
        error: response.status === 401 ? "TTS auth failed" :
               response.status === 429 ? "TTS rate limited upstream" :
               "TTS generation failed",
      });
    }

    const audioBuffer = Buffer.from(await response.arrayBuffer());

    // 3. Cache for next time (best effort — don't block the response)
    let publicUrl = null;
    try {
      publicUrl = await cacheAudio(safeText, voice, audioBuffer);
    } catch {}

    if (publicUrl) {
      return res.json({ url: publicUrl, cached: false });
    }

    // 4. Cache failed — stream the audio directly
    res.set("Content-Type", "audio/mpeg");
    return res.send(audioBuffer);
  } catch (error) {
    if (error?.name === "AbortError") {
      return res.status(504).json({ error: "TTS request timed out" });
    }
    console.error("[tts] error:", error?.message || error);
    return res.status(500).json({ error: "TTS generation failed" });
  } finally {
    clearTimeout(timeoutId);
  }
});

// ─── Postcard image generation (Replicate Flux + Supabase cache) ─────
// Generates a mid-century travel poster (or modernist linework) of a city
// landmark. Image is cached in Supabase Storage so subsequent requests
// for the same {city, style} pair return instantly without re-generating.
//
// Required env:
//   REPLICATE_API_TOKEN — required for first-time generation
//   SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY — required for caching
//
// If REPLICATE_API_TOKEN is not set, the endpoint returns 503 and the
// client falls back to programmatic SVG art (we already built that). So
// the app keeps working even before this env is configured.

const POSTCARD_BUCKET = "postcards-cache"; // Public bucket in Supabase Storage
const POSTCARD_PROMPT_VERSION = "v5"; // Bump to invalidate all cached postcards

// Tier descriptors — drive the time-of-day + atmosphere progression as the
// user levels up in a city. The narrative: as you deepen your relationship
// with a city, you watch it from noon (level 1, fresh arrival) to midnight
// (level 8, you know its secrets). Level 7-8 also get programmatic foil
// embellishments client-side (not part of the AI prompt — those are
// rendered by LimitedEditionOverlay on the client to keep them consistent).
const TIER_DESCRIPTORS = {
  1: "at high noon under bright clear daylight, simple flat 4-color palette, clean confident Risograph poster",
  2: "in the late morning, with warm sunlight and soft long shadows, slightly painterly poster style",
  3: "at golden hour with saturated warm coral and amber light, long shadows raking across the scene, glowing atmosphere",
  4: "at dramatic sunset, the sky deep amber and rose, the landmark silhouetted against the warm horizon, painterly",
  5: "at twilight blue hour with atmospheric mist, deep navy sky transitioning to gold near the horizon, mysterious quiet mood",
  6: "at night under a starry sky, the landmark illuminated by golden warm uplighting, distant city lights twinkling, magical atmosphere",
  7: "at night with the landmark dramatically illuminated, deep navy sky with stars, refined cinematic composition, fine art poster quality",
  8: "at deep night under aurora-like atmosphere, the landmark illuminated by ethereal golden light from below, ultra-detailed, museum-quality fine art poster, awe-inspiring monumental composition",
};

// Per-city prompt fragments. We constrain style tightly so all generated
// art reads as the same series. Cities not in this map fall back to a
// generic prompt — those will look generic; they're best with bespoke text.
const CITY_PROMPTS = {
  paris:      { landmark: "the Eiffel Tower in Paris France, its iconic four arched iron legs visible at the base, with traditional Haussmannian Parisian rooftops below. European architecture only. NOT a Japanese pagoda, NOT cherry blossoms.", palette: "soft peach, golden sun yellow, dark teal forest green, cream" },
  tokyo:      { landmark: "a traditional Japanese five-story wooden pagoda temple (like Senso-ji or Kiyomizu-dera), with red and white painted wood, curved upturned eaves, ornamental finial spire at top. NOT Tokyo Tower, NOT a metal observation tower, NOT the Eiffel Tower. With cherry blossom branches in the foreground and a rising sun behind.", palette: "deep coral pink, ivory cream, bright sun yellow, dark plum" },
  "new york": { landmark: "the Manhattan skyline featuring the Empire State Building and the Chrysler Building with its tiered art-deco crown", palette: "warm amber, deep navy, cream, sunset orange" },
  lisbon:     { landmark: "the Belém Tower with crenellated walls and watchtower keep, the Tagus river in the foreground", palette: "warm gold, terracotta, deep slate blue, cream" },
  barcelona:  { landmark: "the four iconic spires of Sagrada Familia with their pinecone-shaped Gaudí finials", palette: "coral red, warm gold, deep umber brown, cream" },
  istanbul:   { landmark: "the Hagia Sophia with its central dome, side half-domes, and four tall minarets framing it", palette: "warm gold, deep navy blue, terracotta, cream" },
  kyoto:      { landmark: "a path of red torii gates receding into the distance up a mountain at Fushimi Inari shrine", palette: "muted vermilion, soft cream, mountain green, deep brown" },
  rome:       { landmark: "the Colosseum's iconic oval silhouette with three tiers of arches and the ruined section on one side", palette: "warm golden hour amber, terracotta, deep umber, cream" },
  london:     { landmark: "Big Ben tower with its illuminated clock face and the gothic Houses of Parliament", palette: "moody grey blue, warm cream, amber gold, deep slate" },
  dubai:      { landmark: "the Burj Khalifa rising from desert dunes, its tapered stepped silhouette piercing the sky", palette: "warm desert gold, cream, deep amber, soft rose" },
};

function clampLevel(level) {
  const n = Number(level);
  if (!Number.isFinite(n)) return 1;
  return Math.max(1, Math.min(8, Math.round(n)));
}

// Clients used to send "Kyoto, Japan" for the same city others sent as
// "Kyoto", which produced two cache objects and two paid Replicate
// generations for one city — and missed the curated CITY_PROMPTS entry,
// silently returning generic art. The client now normalizes before
// sending, but older installs don't, so the server canonicalizes too.
function bareCityName(city) {
  return String(city || "").split(",")[0].trim();
}

function postcardCacheKey(city, style, level) {
  const cleanCity = bareCityName(city).toLowerCase().replace(/[^a-z0-9-]/g, "-");
  const cleanStyle = style === "modernist" ? "modernist" : "poster";
  const cleanLevel = clampLevel(level);
  // .png because flux-1.1-pro outputs PNG (schnell was WebP). Bumping the
  // version invalidates all previous .webp cache entries automatically.
  return `${POSTCARD_PROMPT_VERSION}/${cleanCity}-L${cleanLevel}-${cleanStyle}.png`;
}

function buildPostcardPrompt(city, style, level) {
  const cityEntry = CITY_PROMPTS[city.toLowerCase().trim()] || {
    landmark: `the most iconic landmark of ${city}`,
    palette: "warm sunset gold, terracotta, cream, deep umber",
  };

  const lvl = clampLevel(level);
  const tierMood = TIER_DESCRIPTORS[lvl] || TIER_DESCRIPTORS[1];

  // CRITICAL composition guidance:
  // - "fully visible, centered, generous breathing room above and below"
  //   prevents the AI from cropping the landmark to the top/bottom edge
  // - "horizontal landscape composition" matches the postcard art zone
  //   (~5:4 aspect) so resizeMode="cover" doesn't crop top/bottom
  // - "landmark occupies the center 60%" gives consistent framing across cities
  const composition = `The landmark is fully visible and centered, with generous breathing room of sky above its top and foreground/horizon below its base. Landmark occupies the center 55-65% of the frame vertically. Horizontal landscape composition, 5:4 aspect ratio. Iconic silhouette readable at a glance.`;

  if (style === "modernist") {
    return `Editorial modernist line illustration of ${cityEntry.landmark}, ${tierMood}. ${composition} Style: single-weight ink linework, minimal flat color washes in ${cityEntry.palette}, generous negative space, magazine cover aesthetic, Christoph Niemann meets Saul Bass, clean geometric. NO TEXT, NO LOGOS, NO PEOPLE, NO BORDERS, NO FRAME.`;
  }
  return `Mid-century travel poster illustration of ${cityEntry.landmark}, ${tierMood}. ${composition} Style: bold flat geometric shapes, limited 4-color palette of ${cityEntry.palette}, vintage screen-printed Risograph aesthetic, Hatch Show Print meets Werkbund. NO TEXT, NO LOGOS, NO PEOPLE, NO BORDERS, NO FRAME.`;
}

async function getCachedPostcard(city, style, level) {
  if (!supabase) return null;
  const key = postcardCacheKey(city, style, level);
  try {
    const { data } = supabase.storage.from(POSTCARD_BUCKET).getPublicUrl(key);
    if (!data?.publicUrl) return null;
    const head = await fetch(data.publicUrl, { method: "HEAD" });
    return head.ok ? data.publicUrl : null;
  } catch {
    return null;
  }
}

async function cachePostcard(city, style, level, imageBuffer) {
  if (!supabase) return null;
  const key = postcardCacheKey(city, style, level);
  try {
    const { error } = await supabase.storage
      .from(POSTCARD_BUCKET)
      .upload(key, imageBuffer, {
        contentType: "image/png",
        upsert: true, // overwrite if exists (prompt version bump invalidates)
      });
    if (error && !error.message.includes("already exists")) {
      console.warn("[postcard] cache upload error:", error.message);
      return null;
    }
    const { data } = supabase.storage.from(POSTCARD_BUCKET).getPublicUrl(key);
    return data?.publicUrl ?? null;
  } catch (err) {
    console.warn("[postcard] cache upload threw:", err?.message || err);
    return null;
  }
}

// Replicate model + input shape. Flux 1.1 Pro produces noticeably better
// composition + lighting fidelity than Schnell — worth the ~10x cost for
// a once-per-{city, level, style} cache miss. Generation takes 10-20s
// instead of 3-5s; the unlock cinematic + collection view hide the wait.
const REPLICATE_MODEL = "black-forest-labs/flux-1.1-pro";

async function generatePostcardImage(prompt) {
  const token = process.env.REPLICATE_API_TOKEN;
  if (!token) {
    throw new Error("REPLICATE_API_TOKEN not configured");
  }

  // Step 1: kick off the prediction. Replicate caps `Prefer: wait` at 60s,
  // so we ask for the max and then poll if it's still processing.
  const createRes = await fetch(
    `https://api.replicate.com/v1/models/${REPLICATE_MODEL}/predictions`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
        Prefer: "wait=60",
      },
      body: JSON.stringify({
        input: {
          prompt,
          aspect_ratio: "5:4",
          output_format: "png",
          output_quality: 95,
        },
      }),
    }
  );

  if (!createRes.ok) {
    const text = await createRes.text();
    throw new Error(`Replicate ${createRes.status}: ${text.slice(0, 300)}`);
  }

  let prediction = await createRes.json();

  // Step 2: if still processing after the 60s wait window, poll until done.
  // flux-1.1-pro typically completes within the initial wait, but on busy
  // hours can take longer. Total wait cap = ~150s.
  const POLL_INTERVAL_MS = 1500;
  const POLL_DEADLINE = Date.now() + 90_000; // 90s additional poll budget

  while (
    prediction.status !== "succeeded" &&
    prediction.status !== "failed" &&
    prediction.status !== "canceled" &&
    Date.now() < POLL_DEADLINE
  ) {
    await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
    const pollRes = await fetch(prediction.urls?.get || `https://api.replicate.com/v1/predictions/${prediction.id}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!pollRes.ok) {
      const text = await pollRes.text();
      throw new Error(`Replicate poll ${pollRes.status}: ${text.slice(0, 200)}`);
    }
    prediction = await pollRes.json();
  }

  if (prediction.status !== "succeeded") {
    throw new Error(
      `Replicate prediction did not succeed: status=${prediction.status} error=${prediction.error || "(none)"}`
    );
  }

  const output = Array.isArray(prediction.output) ? prediction.output[0] : prediction.output;
  if (typeof output !== "string") {
    throw new Error("Replicate returned no output URL");
  }

  // Step 3: download the image so we can re-host it in Supabase.
  // Replicate URLs are time-limited; ours are permanent.
  const imgRes = await fetch(output);
  if (!imgRes.ok) throw new Error(`Failed to download generated image: ${imgRes.status}`);
  const arrayBuf = await imgRes.arrayBuffer();
  return Buffer.from(arrayBuf);
}

app.post("/postcard-image", aiLimiter, requireAuth, async (req, res) => {
  const { city, style, forceFresh, level } = req.body ?? {};
  if (!city || typeof city !== "string" || !city.trim()) {
    return res.status(400).json({ error: "city is required" });
  }
  const styleNormalized = style === "modernist" ? "modernist" : "poster";
  const levelNormalized = clampLevel(level);

  try {
    // 1) Cache hit? (skipped when caller requests a fresh generation —
    //    the "Regenerate" button or first-time regen after a bad output)
    if (!forceFresh) {
      const cached = await getCachedPostcard(city, styleNormalized, levelNormalized);
      if (cached) {
        return res.json({ url: cached, cached: true });
      }
    }

    // 2) Generate
    if (!process.env.REPLICATE_API_TOKEN) {
      return res.status(503).json({
        error: "Image generation not configured (REPLICATE_API_TOKEN missing)",
      });
    }

    const prompt = buildPostcardPrompt(city, styleNormalized, levelNormalized);
    const imageBuffer = await generatePostcardImage(prompt);

    // 3) Cache and return
    const publicUrl = await cachePostcard(city, styleNormalized, levelNormalized, imageBuffer);
    if (publicUrl) {
      return res.json({ url: publicUrl, cached: false });
    }

    // Cache failed — stream the bytes directly (caller can use them once).
    res.set("Content-Type", "image/webp");
    return res.send(imageBuffer);
  } catch (err) {
    console.error("[postcard] error:", err?.message || err);
    if (err?.message?.includes("REPLICATE_API_TOKEN")) {
      return res.status(503).json({ error: "Image generation not configured" });
    }
    return res.status(500).json({ error: "Postcard generation failed" });
  }
});

const PORT = process.env.PORT || 10000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
