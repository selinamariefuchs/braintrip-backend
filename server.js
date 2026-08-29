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
  "claude-sonnet-4-6",
  "claude-sonnet-4-20250514",
  "claude-haiku-4-5-20251001",
  "claude-haiku-4-5",
]);

// Models that were allowed once and have since been retired upstream.
//
// Scan sends "claude-sonnet-4-20250514", which passed the allowlist and then
// came back 404 not_found_error from Anthropic — a break that looked like a
// scan bug and was really a dead model id. The version on the App Store has
// that id compiled into it, so removing it here would leave every installed
// copy broken until they update. Remapping fixes them where they stand.
const ANTHROPIC_MODEL_ALIASES = {
  "claude-sonnet-4-20250514": "claude-sonnet-4-6",
};

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
const PHOTO_REF_CACHE = new Map(); // `${width}:${ref}` -> googleusercontent url

app.get("/photo-ref", globalLimiter, async (req, res) => {
  const key = process.env.GOOGLE_PLACES_API_KEY;
  if (!key) return res.status(503).json({ error: "Places not configured" });
  // Google's photo references have no documented maximum and vary widely —
  // measured 431 to 650 characters across one Chicago search. The old 600
  // cap silently truncated the longest ones into invalid references, which
  // came back as a 404 and rendered as a grey square. Still bounded, just
  // above anything Google actually sends.
  const ref = String(req.query.ref || "").trim().slice(0, 4000);
  if (!ref) return res.status(400).json({ error: "ref is required" });

  // How wide the caller actually needs the image.
  //
  // This was hard-coded at 640 for everything, while the two callers that use
  // it are 56pt and 44pt thumbnails — about 168px on a 3x screen. Every card
  // was pulling roughly 110KB to fill a box needing a fraction of that, and
  // the Near list mounts every card at once, so a single screen cost several
  // megabytes before anything appeared. Callers now ask for what they need.
  //
  // 640 stays the default so any caller that does not pass w is unaffected.
  const requested = Number.parseInt(String(req.query.w || ""), 10);
  const width = Number.isFinite(requested)
    ? Math.min(Math.max(requested, 64), 1600)
    : 640;

  // The cache key has to carry the width. It used to be the ref alone, which
  // was correct while there was one width and silently wrong the moment there
  // wasn't: a thumbnail request would have been served whatever width landed
  // in the cache first.
  const cacheKey = `${width}:${ref}`;
  const cached = PHOTO_REF_CACHE.get(cacheKey);
  if (cached) return res.redirect(302, cached);
  try {
    const photoRes = await fetch(
      `https://maps.googleapis.com/maps/api/place/photo?maxwidth=${width}&photo_reference=${encodeURIComponent(ref)}&key=${key}`,
      { redirect: "manual" }
    );
    const loc = photoRes.headers.get("location");
    if (!loc) return res.status(404).json({ error: "No photo" });
    PHOTO_REF_CACHE.set(cacheKey, loc);
    return res.redirect(302, loc);
  } catch (err) {
    return res.status(500).json({ error: "Photo lookup failed" });
  }
});

// ─── Nearby landmark (daily photo challenge) ────────────────
// Finds the most prominent tourist attraction within ~2km of the given
// coordinates. Cached per ~1km grid cell — landmarks don't move.
// ─── Around me: real nearby places, by category and radius ──────────────
//
// Every result here comes from Google Places. Nothing is model-generated,
// deliberately: a landmark that doesn't exist is a bad recommendation, but
// an invented boat rental is someone driving to a pier for a business that
// was never there. Activities are the category most worth having and the
// least safe to hallucinate, so they are sourced, not written.
//
// Legacy Nearby Search takes a single type per call, and has no type at all
// for things like boat hire or carriage rides — so each category fans out
// across several typed and keyword searches and the results are merged on
// place_id.
// What there is to do, as a traveller would name it.
//
// `idea` is the thing being chosen between. `commodity` marks the ones where
// the vendors are interchangeable — you want a boat, or pizza, not a
// particular operator — so the app can collapse them into one row. A museum
// is not interchangeable: there is one, and it is the destination.
//
// These are deliberately long and served a page at a time. Every entry is a
// separate Google call, so running all of them on first paint would be slow
// and expensive; the app asks for the next page when the reader reaches the
// end of the list, which is also the only point at which they are wanted.
const AROUND_CATEGORIES = {
  attractions: [
    { type: "museum", idea: "Museum" },
    { type: "tourist_attraction", idea: "Attraction" },
    { type: "art_gallery", idea: "Gallery" },
    { type: "park", idea: "Park" },
    { keyword: "observation deck", idea: "Observation deck" },
    { keyword: "historic landmark", idea: "Historic landmark" },
    { keyword: "botanical garden", idea: "Botanical garden" },
    { keyword: "monument", idea: "Monument" },
    { keyword: "cathedral", idea: "Cathedral" },
    { keyword: "scenic viewpoint", idea: "Viewpoint" },
    { type: "library", idea: "Library" },
    { keyword: "public art", idea: "Public art" },
    { keyword: "historic theatre", idea: "Historic theatre" },
    { keyword: "market hall", idea: "Market hall" },
  ],
  activities: [
    { keyword: "boat rental", idea: "Rent a boat", commodity: true },
    { keyword: "bike rental", idea: "Rent a bike", commodity: true },
    { keyword: "guided walking tour", idea: "Walking tour", commodity: true },
    { keyword: "boat tour", idea: "Boat tour", commodity: true },
    { keyword: "kayak rental", idea: "Rent a kayak", commodity: true },
    { keyword: "segway tour", idea: "Segway tour", commodity: true },
    { keyword: "brewery tour", idea: "Brewery tour", commodity: true },
    { keyword: "food tour", idea: "Food tour", commodity: true },
    { keyword: "escape room", idea: "Escape room", commodity: true },
    { keyword: "comedy club", idea: "Comedy show", commodity: true },
    { keyword: "live music venue", idea: "Live music", commodity: true },
    { keyword: "rooftop bar", idea: "Rooftop bar", commodity: true },
    { keyword: "cooking class", idea: "Cooking class", commodity: true },
    { type: "spa", idea: "Spa", commodity: true },
    { type: "bowling_alley", idea: "Bowling", commodity: true },
    { type: "movie_theater", idea: "Cinema", commodity: true },
    { keyword: "mini golf", idea: "Mini golf", commodity: true },
    { keyword: "arcade", idea: "Arcade", commodity: true },
    { keyword: "helicopter tour", idea: "Helicopter tour", commodity: true },
    { keyword: "horse carriage ride", idea: "Carriage ride", commodity: true },
    // These four are commodity in the sense that matters here: in "Things to
    // Do" every row should be something to go and do, with the venues behind
    // a tap. Without the flag they came through as bare business names in a
    // list of ideas — Google files The Great Escape Room and Color Factory
    // under amusement_park, so the activities list read "Boat tour, Helicopter
    // tour, The Great Escape Room Chicago, Carriage ride", which is the exact
    // stray-business problem the grouping exists to prevent.
    //
    // Attractions is where a singular destination stays itself: the Art
    // Institute must not hide inside "Museums (3)". That distinction is by
    // category, not by venue type, which is why the same idea can be grouped
    // here and standalone there.
    { type: "amusement_park", idea: "Amusement park", commodity: true },
    { type: "aquarium", idea: "Aquarium", commodity: true },
    { type: "zoo", idea: "Zoo", commodity: true },
    { type: "stadium", idea: "Stadium", commodity: true },
  ],
  // Food groups by what you feel like eating, which is the actual decision.
  // "Restaurant" as one row containing forty places would be no better than
  // the flat list it replaced.
  food: [
    { keyword: "coffee", idea: "Coffee", commodity: true },
    { keyword: "pizza", idea: "Pizza", commodity: true },
    { keyword: "brunch", idea: "Brunch", commodity: true },
    { keyword: "sushi", idea: "Sushi", commodity: true },
    { keyword: "tacos", idea: "Tacos", commodity: true },
    { keyword: "burgers", idea: "Burgers", commodity: true },
    { keyword: "ramen", idea: "Ramen", commodity: true },
    { keyword: "bakery", idea: "Bakery", commodity: true },
    { keyword: "cocktail bar", idea: "Cocktails", commodity: true },
    { keyword: "brewery", idea: "Brewery", commodity: true },
    { keyword: "seafood", idea: "Seafood", commodity: true },
    { keyword: "steakhouse", idea: "Steakhouse", commodity: true },
    { keyword: "thai food", idea: "Thai", commodity: true },
    { keyword: "italian restaurant", idea: "Italian", commodity: true },
    { keyword: "barbecue", idea: "Barbecue", commodity: true },
    { keyword: "ice cream", idea: "Ice cream", commodity: true },
    { keyword: "vegetarian restaurant", idea: "Vegetarian", commodity: true },
    { keyword: "wine bar", idea: "Wine bar", commodity: true },
  ],
};

// What this particular city is for.
//
// The generic list is the same in Reykjavik as in Bangkok, which is exactly
// what a travel app should not feel like. These lead the first page for the
// city you are standing in, so Chicago opens with deep dish and an
// architecture boat tour rather than "Rent a bike".
//
// They are keywords, not content: the results still come from Google, so an
// idea that finds nothing simply doesn't appear. Nothing here can invent a
// restaurant.
const CITY_IDEAS = {
  chicago: [
    { keyword: "deep dish pizza", idea: "Deep dish", commodity: true, category: "food" },
    { keyword: "italian beef sandwich", idea: "Italian beef", commodity: true, category: "food" },
    // Chicago-style hot dogs and caramel-and-cheese popcorn are as particular
    // to this city as deep dish, and a "burgers" or "bakery" search finds
    // neither of them.
    { keyword: "chicago style hot dog", idea: "Chicago dog", commodity: true, category: "food" },
    { keyword: "caramel cheese popcorn", idea: "Chicago mix", commodity: true, category: "food" },
    { keyword: "blues club", idea: "Blues club", commodity: true, category: "activities" },
    { keyword: "architecture river cruise", idea: "Architecture cruise", commodity: true, category: "activities" },
    // Improv was invented here, and the jazz and Prohibition rooms are the
    // other two things people come to Chicago for at night. The generic list
    // has "comedy club" and "live music venue", which find neither Second
    // City nor the Green Mill for what they actually are.
    { keyword: "improv comedy theater", idea: "Improv comedy", commodity: true, category: "activities" },
    { keyword: "jazz club", idea: "Jazz club", commodity: true, category: "activities" },
    { keyword: "speakeasy bar", idea: "Speakeasy", commodity: true, category: "activities" },
    // The lakefront is the thing residents would name first and no generic
    // "park" search surfaces as what it is.
    { keyword: "lakefront beach", idea: "Lake beach", category: "attractions" },
    { keyword: "riverwalk", idea: "Riverwalk", category: "attractions" },
  ],
  tokyo: [
    { keyword: "ramen", idea: "Ramen", commodity: true, category: "food" },
    { keyword: "izakaya", idea: "Izakaya", commodity: true, category: "food" },
    { keyword: "conveyor belt sushi", idea: "Conveyor sushi", commodity: true, category: "food" },
    { keyword: "karaoke", idea: "Karaoke", commodity: true, category: "activities" },
  ],
  kyoto: [
    { keyword: "matcha tea house", idea: "Matcha house", commodity: true, category: "food" },
    { keyword: "kaiseki", idea: "Kaiseki", commodity: true, category: "food" },
    { keyword: "temple garden", idea: "Temple garden", category: "attractions" },
    { keyword: "kimono rental", idea: "Rent a kimono", commodity: true, category: "activities" },
  ],
  paris: [
    { keyword: "patisserie", idea: "Patisserie", commodity: true, category: "food" },
    { keyword: "bistro", idea: "Bistro", commodity: true, category: "food" },
    { keyword: "wine cave tasting", idea: "Wine tasting", commodity: true, category: "activities" },
  ],
  "new york": [
    { keyword: "bagels", idea: "Bagels", commodity: true, category: "food" },
    { keyword: "pizza slice", idea: "Slice shop", commodity: true, category: "food" },
    { keyword: "jazz club", idea: "Jazz club", commodity: true, category: "activities" },
    { keyword: "jewish deli", idea: "Deli", commodity: true, category: "food" },
  ],
  lisbon: [
    { keyword: "pastel de nata", idea: "Pastel de nata", commodity: true, category: "food" },
    { keyword: "fado house", idea: "Fado", commodity: true, category: "activities" },
    { keyword: "miradouro viewpoint", idea: "Miradouro", category: "attractions" },
  ],
  barcelona: [
    { keyword: "tapas", idea: "Tapas", commodity: true, category: "food" },
    { keyword: "vermouth bar", idea: "Vermut", commodity: true, category: "food" },
    { keyword: "gaudi building", idea: "Gaudí building", category: "attractions" },
  ],
  rome: [
    { keyword: "gelato", idea: "Gelato", commodity: true, category: "food" },
    { keyword: "trattoria", idea: "Trattoria", commodity: true, category: "food" },
    { keyword: "roman ruins", idea: "Roman ruins", category: "attractions" },
  ],
  london: [
    { keyword: "pub", idea: "Pub", commodity: true, category: "food" },
    { keyword: "afternoon tea", idea: "Afternoon tea", commodity: true, category: "food" },
    { keyword: "curry house", idea: "Curry house", commodity: true, category: "food" },
  ],
  istanbul: [
    { keyword: "turkish bath hammam", idea: "Hammam", commodity: true, category: "activities" },
    { keyword: "meze restaurant", idea: "Meze", commodity: true, category: "food" },
    { keyword: "turkish coffee", idea: "Turkish coffee", commodity: true, category: "food" },
  ],
  bangkok: [
    { keyword: "street food", idea: "Street food", commodity: true, category: "food" },
    { keyword: "thai massage", idea: "Thai massage", commodity: true, category: "activities" },
    { keyword: "night market", idea: "Night market", commodity: true, category: "activities" },
  ],
  "mexico city": [
    { keyword: "tacos al pastor", idea: "Al pastor", commodity: true, category: "food" },
    { keyword: "mezcaleria", idea: "Mezcal bar", commodity: true, category: "food" },
    { keyword: "mercado", idea: "Mercado", commodity: true, category: "activities" },
  ],
  "buenos aires": [
    { keyword: "parrilla", idea: "Parrilla", commodity: true, category: "food" },
    { keyword: "tango show", idea: "Tango", commodity: true, category: "activities" },
    { keyword: "empanadas", idea: "Empanadas", commodity: true, category: "food" },
  ],
  marrakech: [
    { keyword: "hammam", idea: "Hammam", commodity: true, category: "activities" },
    { keyword: "tagine restaurant", idea: "Tagine", commodity: true, category: "food" },
    { keyword: "souk", idea: "Souk", commodity: true, category: "activities" },
  ],
  amsterdam: [
    { keyword: "brown cafe", idea: "Brown café", commodity: true, category: "food" },
    { keyword: "canal cruise", idea: "Canal cruise", commodity: true, category: "activities" },
    { keyword: "stroopwafel", idea: "Stroopwafel", commodity: true, category: "food" },
  ],
  berlin: [
    { keyword: "currywurst", idea: "Currywurst", commodity: true, category: "food" },
    { keyword: "biergarten", idea: "Biergarten", commodity: true, category: "food" },
    { keyword: "doner kebab", idea: "Döner", commodity: true, category: "food" },
  ],
  reykjavik: [
    { keyword: "geothermal pool", idea: "Geothermal pool", commodity: true, category: "activities" },
    { keyword: "hot dog stand", idea: "Hot dog stand", commodity: true, category: "food" },
    { keyword: "seafood soup", idea: "Seafood soup", commodity: true, category: "food" },
  ],
  "cape town": [
    { keyword: "wine tasting", idea: "Wine tasting", commodity: true, category: "activities" },
    { keyword: "braai", idea: "Braai", commodity: true, category: "food" },
  ],
  "rio de janeiro": [
    { keyword: "churrascaria", idea: "Churrascaria", commodity: true, category: "food" },
    { keyword: "acai bowl", idea: "Açaí", commodity: true, category: "food" },
    { keyword: "samba club", idea: "Samba", commodity: true, category: "activities" },
  ],
  sydney: [
    { keyword: "flat white coffee", idea: "Flat white", commodity: true, category: "food" },
    { keyword: "coastal walk", idea: "Coastal walk", category: "attractions" },
    { keyword: "seafood market", idea: "Seafood market", commodity: true, category: "food" },
  ],
  singapore: [
    { keyword: "hawker centre", idea: "Hawker centre", commodity: true, category: "food" },
    { keyword: "chilli crab", idea: "Chilli crab", commodity: true, category: "food" },
    { keyword: "kaya toast", idea: "Kaya toast", commodity: true, category: "food" },
  ],
  mumbai: [
    { keyword: "street chaat", idea: "Chaat", commodity: true, category: "food" },
    { keyword: "thali restaurant", idea: "Thali", commodity: true, category: "food" },
    { keyword: "irani cafe", idea: "Irani café", commodity: true, category: "food" },
  ],
  dubai: [
    { keyword: "shawarma", idea: "Shawarma", commodity: true, category: "food" },
    { keyword: "desert safari", idea: "Desert safari", commodity: true, category: "activities" },
    { keyword: "gold souk", idea: "Souk", commodity: true, category: "activities" },
  ],
  bali: [
    { keyword: "warung", idea: "Warung", commodity: true, category: "food" },
    { keyword: "beach club", idea: "Beach club", commodity: true, category: "activities" },
    { keyword: "yoga studio", idea: "Yoga", commodity: true, category: "activities" },
  ],
  phuket: [
    { keyword: "thai massage", idea: "Thai massage", commodity: true, category: "activities" },
    { keyword: "beach club", idea: "Beach club", commodity: true, category: "activities" },
    { keyword: "night market", idea: "Night market", commodity: true, category: "activities" },
  ],
  santorini: [
    { keyword: "greek taverna", idea: "Taverna", commodity: true, category: "food" },
    { keyword: "sunset viewpoint", idea: "Sunset spot", category: "attractions" },
    { keyword: "winery tasting", idea: "Winery", commodity: true, category: "activities" },
  ],
};

/** The city's own ideas for this category, if we know the city. */
function cityIdeasFor(city, category) {
  if (!city) return [];
  const key = String(city).trim().toLowerCase();
  const all = CITY_IDEAS[key];
  if (!Array.isArray(all)) return [];
  return all.filter((i) => i.category === category);
}

/** Searches run per request. More ideas cost more Google calls and latency. */
const AROUND_PAGE_SIZE = 6;

const AROUND_CACHE = new Map(); // cell|radius|category -> payload
const AROUND_CACHE_TTL_MS = 30 * 60 * 1000;

function metersBetween(lat1, lng1, lat2, lng2) {
  const R = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLng / 2) ** 2;
  return Math.round(2 * R * Math.asin(Math.sqrt(a)));
}

app.get("/around-me", globalLimiter, async (req, res) => {
  const key = process.env.GOOGLE_PLACES_API_KEY;
  if (!key) return res.status(503).json({ error: "Places not configured" });

  const lat = Number(req.query.lat);
  const lng = Number(req.query.lng);
  const radius = Math.max(500, Math.min(50000, Number(req.query.radius) || 2000));
  const category = String(req.query.category || "attractions");
  // The reverse-geocoded city, so the list can lead with what this place is
  // actually known for.
  const city = String(req.query.city || "").trim().slice(0, 80);
  const generic = AROUND_CATEGORIES[category];
  // City ideas go first: they are the reason to open the app here rather
  // than a maps app, so they should not be on page three.
  const allQueries = Array.isArray(generic)
    ? [...cityIdeasFor(city, category), ...generic]
    : generic;
  // Which slice of the idea list to run. The reader asks for the next one on
  // reaching the end of what they have.
  const page = Math.max(0, Math.min(20, Number(req.query.page) || 0));
  const queries = Array.isArray(allQueries)
    ? allQueries.slice(page * AROUND_PAGE_SIZE, (page + 1) * AROUND_PAGE_SIZE)
    : allQueries;
  const hasMore = Array.isArray(allQueries)
    ? (page + 1) * AROUND_PAGE_SIZE < allQueries.length
    : false;

  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    return res.status(400).json({ error: "lat and lng are required" });
  }
  if (!generic) {
    return res.status(400).json({
      error: "Unknown category",
      allowed: Object.keys(AROUND_CATEGORIES),
    });
  }

  // ~1km cell: fine enough that results stay relevant, coarse enough that
  // walking around doesn't re-bill every Places call.
  const cacheKey =
    `${lat.toFixed(2)},${lng.toFixed(2)}|${radius}|${category}|${page}|${city.toLowerCase()}`;
  const hit = AROUND_CACHE.get(cacheKey);
  if (hit && Date.now() - hit.at < AROUND_CACHE_TTL_MS) {
    return res.json({ ...hit.payload, cached: true });
  }

  try {
    const settled = await Promise.allSettled(
      queries.map((q) => {
        const param = q.type
          ? `type=${encodeURIComponent(q.type)}`
          : `keyword=${encodeURIComponent(q.keyword)}`;
        return fetch(
          `https://maps.googleapis.com/maps/api/place/nearbysearch/json` +
            `?location=${lat},${lng}&radius=${radius}&${param}&key=${key}`
        ).then((r) => r.json());
      })
    );

    const byId = new Map();
    // Which search produced each result. This was being discarded, which is
    // why six boat companies arrived as six unrelated rows — the grouping
    // the app wants was already in the request and thrown away on the way out.
    for (let qi = 0; qi < settled.length; qi++) {
      const outcome = settled[qi];
      const q = queries[qi] || {};
      if (outcome.status !== "fulfilled") continue;
      for (const r of outcome.value?.results || []) {
        if (!r?.place_id || byId.has(r.place_id)) continue;
        if (r.business_status && r.business_status !== "OPERATIONAL") continue;
        const plat = r.geometry?.location?.lat;
        const plng = r.geometry?.location?.lng;
        if (!Number.isFinite(plat) || !Number.isFinite(plng)) continue;
        byId.set(r.place_id, {
          placeId: r.place_id,
          name: r.name,
          vicinity: r.vicinity || r.formatted_address || "",
          rating: typeof r.rating === "number" ? r.rating : null,
          totalRatings: r.user_ratings_total || 0,
          priceLevel: typeof r.price_level === "number" ? r.price_level : null,
          photoRef: r.photos?.[0]?.photo_reference || null,
          lat: plat,
          lng: plng,
          distanceMeters: metersBetween(lat, lng, plat, plng),
          openNow: r.opening_hours?.open_now ?? null,
          idea: q.idea || null,
          commodity: q.commodity === true,
        });
      }
    }

    // A handful of ratings is the difference between a real destination and
    // somebody's mislabelled shopfront.
    const places = Array.from(byId.values())
      .filter((p) => p.totalRatings >= 5)
      .sort((a, b) => a.distanceMeters - b.distanceMeters)
      // Generous: the app groups interchangeable vendors into one row, so a
      // large number of results becomes a small number of ideas.
      .slice(0, 80);

    const payload = {
      category, radius, page, hasMore, count: places.length, places,
      cityIdeas: cityIdeasFor(city, category).map((i) => i.idea),
    };
    AROUND_CACHE.set(cacheKey, { at: Date.now(), payload });
    return res.json(payload);
  } catch (err) {
    console.warn("[around-me] failed:", err?.stack || err);
    return res.status(500).json({
      error: "Lookup failed",
      reason: String(err?.message || err).slice(0, 200),
    });
  }
});

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
      `https://maps.googleapis.com/maps/api/place/details/json?place_id=${encodeURIComponent(placeId)}&fields=name,rating,user_ratings_total,reviews,editorial_summary,types,url,website,formatted_address,price_level,current_opening_hours&reviews_sort=most_relevant&key=${key}`
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

    // Every field has to be named here as well as in the request — this
    // object IS the response, so anything omitted is silently dropped no
    // matter what Google sent. That is how `place` went missing from trivia.
    const payload = {
      name: typeof result.name === "string" ? result.name : null,
      rating: typeof result.rating === "number" ? result.rating : null,
      totalRatings: typeof result.user_ratings_total === "number" ? result.user_ratings_total : null,
      // Google's own one-line description of the place, where it has one.
      summary: typeof result.editorial_summary?.overview === "string"
        ? result.editorial_summary.overview
        : null,
      types: Array.isArray(result.types) ? result.types : [],
      address: typeof result.formatted_address === "string" ? result.formatted_address : null,
      priceLevel: typeof result.price_level === "number" ? result.price_level : null,
      openNow: result.current_opening_hours?.open_now ?? null,
      hoursToday: Array.isArray(result.current_opening_hours?.weekday_text)
        ? result.current_opening_hours.weekday_text[(new Date().getDay() + 6) % 7] || null
        : null,
      website: typeof result.website === "string" ? result.website : null,
      /** The canonical Google Maps page for this place. */
      mapsUrl: typeof result.url === "string" ? result.url : null,
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

- Each question must name the real, physical place it concerns in "place" —
  the exact name a visitor would see on a sign or find in Maps ("Sagrada
  Família", "Fushimi Inari Taisha", "Billy Goat Tavern"). This is how the
  app tells someone standing nearby that they already learned about it.
- If a question is about the city as a whole and no single place fits, set
  "place" to an empty string rather than inventing somewhere.

Return JSON format:
[
  {
    "question": "...",
    "options": ["A", "B", "C", "D"],
    "correctAnswer": "...",
    "funFact": "...",
    "place": "Sagrada Família"
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
                    place: { type: "string" },
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
                    "place",
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
          // The normaliser rebuilds each question from scratch, so anything
          // not named here is silently dropped on the way out — which is how
          // place went missing between the schema and the response.
          place: typeof q.place === "string" ? q.place.trim() : "",
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
      : "Mix iconic landmarks, local food spots, and lesser known finds equally.";

    const prompt = `
Generate 5 must-visit spots for ${cleanCity}.
${categoryPrompt}

Requirements:
- Each spot must be a real specific place
- For whyGo, write a condensed summary of what real visitors consistently praise — the specific things reviewers actually mention (a dish, a view, a time of day, a detail). Written like a distilled Google-reviews overview, e.g. "Visitors rave about the custard tarts straight from the oven and say the line moves faster than it looks" — NEVER generic praise like "a must-see with something for everyone"
- Mix well known spots with lesser known ones
- hiddenGem: true when a place is genuinely off the usual tourist trail — somewhere a local would send a friend, not something on the first page of every guidebook. This is independent of type: a backstreet restaurant is Food & Drink AND a hidden gem. Of 5 spots, expect roughly 1-2 to be true. Never mark a famous landmark true.
- Keep descriptions fun and engaging
- Never include fake or generic places
- For each spot, include the neighborhood or district name (e.g. "Asakusa", "Shibuya", "Le Marais")
- For each spot, rate the price level: 1 = free, 2 = budget ($1-15), 3 = moderate ($15-50), 4 = expensive ($50+)
- ticketsNeeded: true ONLY when admission itself requires buying a ticket or booking a timed entry slot in advance — museums with timed entry, observation decks, guided tours, theatre or shows, attractions that sell out. A place you can simply walk into is false, no matter how popular or expensive it is.
- reservationRecommended: true when it is a restaurant, bar or cafe where booking a table ahead is genuinely advisable. This is NOT a ticket.
- A restaurant is almost never ticketsNeeded. A museum is almost never reservationRecommended. Most parks, streets, markets and viewpoints are false for both.

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
                    // What the place IS. How well-known it is lives in
                    // hiddenGem, because those are different questions and a
                    // single field made the model choose between them — a
                    // backstreet ramen bar had to give up being food to be a
                    // secret.
                    type: {
                      type: "string",
                      enum: [
                        "Landmark",
                        "Food & Drink",
                        "Experience",
                        "Culture",
                      ],
                    },
                    hiddenGem: { type: "boolean" },
                    description: { type: "string" },
                    whyGo: { type: "string" },
                    photoPrompt: { type: "string" },
                    neighborhood: { type: "string" },
                    priceLevel: { type: "number", enum: [1, 2, 3, 4] },
                    ticketsNeeded: { type: "boolean" },
                    reservationRecommended: { type: "boolean" },
                  },
                  required: ["name", "type", "hiddenGem", "description", "whyGo", "photoPrompt", "neighborhood", "priceLevel", "ticketsNeeded", "reservationRecommended"],
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
        type: typeof s.type === "string" ? s.type : "Landmark",
        // This normaliser rebuilds each spot from a fixed field list, so a
        // field missing here is dropped no matter what the schema demands.
        hiddenGem: typeof s.hiddenGem === "boolean" ? s.hiddenGem : false,
        description: typeof s.description === "string" ? s.description : "",
        whyGo: typeof s.whyGo === "string" ? s.whyGo : "",
        photoPrompt: typeof s.photoPrompt === "string" ? s.photoPrompt : "",
        neighborhood: typeof s.neighborhood === "string" ? s.neighborhood : "",
        priceLevel: [1, 2, 3, 4].includes(s.priceLevel) ? s.priceLevel : 1,
        ticketsNeeded: typeof s.ticketsNeeded === "boolean" ? s.ticketsNeeded : false,
        reservationRecommended:
          typeof s.reservationRecommended === "boolean" ? s.reservationRecommended : false,
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

    const upstreamModel = ANTHROPIC_MODEL_ALIASES[model] ?? model;

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
      body: JSON.stringify({ model: upstreamModel, max_tokens: safeMaxTokens, messages }),
      signal: controller.signal,
    });

    // Keep the raw text too. response.json() on a non-JSON body throws or
    // yields something unrecognisable, and an empty error.message with a
    // populated error.type is not a shape Anthropic returns — which suggests
    // what came back may not be from Anthropic at all.
    const rawBody = await response.text();
    let data;
    try {
      data = JSON.parse(rawBody);
    } catch {
      data = null;
    }

    if (!response.ok) {
      // Don't leak Anthropic's internal error details to clients
      // A per-failure id, logged and returned. Searching the host's logs for
      // "Anthropic upstream error" keeps landing on historical entries, and a
      // stale line sent this investigation down two wrong paths already. This
      // makes one specific failure findable.
      const traceId = crypto.randomUUID().slice(0, 8);
      console.error(
        `Anthropic upstream error [${traceId}]:`,
        response.status,
        data?.error?.message
      );
      // A coarse cause, though. "Upstream service error" alone is
      // undiagnosable from outside — the app's whole trivia feature can be
      // down and the only way to tell an expired key from an unknown model
      // is to read Render's logs. This says which, without leaking anything
      // Anthropic returned.
      const upstreamMsg = String(data?.error?.message || "").toLowerCase();
      // Anthropic's error `type` is a fixed enum — authentication_error,
      // invalid_request_error, not_found_error — so it carries no account
      // detail and is safe to pass on. Without it a 400 is unfalsifiable
      // from outside the host.
      const upstreamType = typeof data?.error?.type === "string" ? data.error.type : null;
      // Match against known causes rather than passing the message through.
      // Each entry is a phrase Anthropic uses; the value is what it means for
      // whoever has to fix it. Anything unmatched stays "upstream" and is
      // only in the logs.
      const KNOWN = [
        // A self-imposed spend cap on the Anthropic org reads as
        // "You have reached your specified API usage limits. You will regain
        // access on <date>." It matched nothing in this list, so it surfaced
        // as a bare "upstream" and took a day and several deploys to identify
        // while trivia and Scan were both down. It gets its own reason now:
        // it is not a credit balance, and adding credits does not clear it.
        ["usage limit", "usage_limit"],
        ["spend limit", "usage_limit"],
        ["credit balance", "credit"],
        ["insufficient", "credit"],
        ["quota", "credit"],
        // Anthropic writes the header name with hyphens — "invalid x-api-key"
        // — which the spaced phrase never matched. A bad key normally arrives
        // as a 401 and is classified by status before reaching this table, so
        // this only matters when the status is something else, but that is
        // exactly the case where the table is all there is.
        ["x-api-key", "auth"],
        ["api key", "auth"],
        ["authentication", "auth"],
        ["permission", "auth"],
        ["model", "model"],
        ["max_tokens", "max_tokens"],
        ["messages", "messages"],
        ["rate", "rate_limited"],
        ["overloaded", "overloaded"],
      ];
      const matched = KNOWN.find(([phrase]) => upstreamMsg.includes(phrase));
      const reason =
        response.status === 401 || response.status === 403 ? "auth"
        : response.status === 429 ? "rate_limited"
        : matched ? matched[1]
        : "upstream";
      // The shape of our OWN request, so a failure is diagnosable without
      // reading the host's logs and without echoing anything Anthropic sent.
      // Field names, types and counts only — no message content.
      // The structure of the reply, not its contents: which keys came back
      // and how long the body was. Enough to tell an Anthropic error from an
      // edge proxy's, without echoing anything either of them wrote.
      const replyShape = {
        bodyLength: rawBody.length,
        topLevelKeys: data && typeof data === "object" ? Object.keys(data) : null,
        errorKeys:
          data && typeof data.error === "object" && data.error
            ? Object.keys(data.error)
            : null,
        messageLength:
          typeof data?.error?.message === "string" ? data.error.message.length : null,
        contentType: response.headers.get("content-type"),
      };
      const sentShape = {
        // What actually went upstream, which is what the error is about.
        model: upstreamModel,
        maxTokens: safeMaxTokens,
        maxTokensType: typeof safeMaxTokens,
        messageCount: Array.isArray(messages) ? messages.length : null,
        contentKind: Array.isArray(messages) && messages[0]
          ? (typeof messages[0].content === "string"
              ? "string"
              : Array.isArray(messages[0].content)
                ? `blocks:${messages[0].content.map((b) => b?.type).join(",")}`
                : typeof messages[0].content)
          : null,
      };
      // The raw upstream message used to be echoed here for invalid_request_error,
      // on the reasoning that such an error only ever describes fields we sent.
      // That reasoning was wrong, and the message that proved it is the one it
      // was added to find: "You have reached your specified API usage limits"
      // arrives as invalid_request_error and is account billing state, not
      // request shape. It is gone again now that it has served its purpose —
      // the full message is still logged above against the traceId, which is
      // where it belongs, and `reason` now classifies this case by name.
      return res.status(response.status >= 500 ? 502 : response.status).json({
        traceId,
        reason,
        upstreamType,
        sentShape,
        replyShape,
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
  tokyo:      { landmark: "the five-story vermilion pagoda of Senso-ji temple in Asakusa Tokyo, red and white painted wood with curved upturned eaves and an ornamental finial spire, rising above the temple's tiled roofs with the great red paper lantern of the Kaminarimon gate below it, dense flat city rooftops stretching unbroken to a low distant horizon", palette: "deep coral pink, ivory cream, bright sun yellow, dark plum" },
  "new york": { landmark: "the Manhattan skyline featuring the Empire State Building and the Chrysler Building with its tiered art-deco crown", palette: "warm amber, deep navy, cream, sunset orange" },
  lisbon:     { landmark: "the Belém Tower with crenellated walls and watchtower keep, the Tagus river in the foreground", palette: "warm gold, terracotta, deep slate blue, cream" },
  barcelona:  { landmark: "the four iconic spires of Sagrada Familia with their pinecone-shaped Gaudí finials", palette: "coral red, warm gold, deep umber brown, cream" },
  istanbul:   { landmark: "the Hagia Sophia with its central dome, side half-domes, and four tall minarets framing it", palette: "warm gold, deep navy blue, terracotta, cream" },
  kyoto:      { landmark: "a dense tunnel of red-orange torii gates along a narrow stone stairway at Fushimi Inari shrine in Kyoto, the gates filling the frame edge to edge and receding into darkness, cedar and bamboo pressing close on both sides beneath a closed forest canopy", palette: "muted vermilion, soft cream, cedar forest green, deep brown" },
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

// Per-city cache version. Bump one city here when only its prompt changed,
// instead of moving POSTCARD_PROMPT_VERSION and throwing away every other
// city's art (each discarded image is a paid regeneration).
//   kyoto — v5 put Mount Fuji behind the gates; v6's NOT clauses failed
//   (diffusion models don't honour negation); v7 closed the canopy but was
//   verbose enough to override the tier mood; v8 trims it back.
const CITY_PROMPT_VERSION = {
  tokyo: "v2", // v1 named Kiyomizu-dera (a Kyoto temple) and drew Kyoto
  kyoto: "v8", // v7 lost the level-8 night; trimmed so the tier descriptor carries
};

function postcardCacheKey(city, style, level) {
  const cleanCity = bareCityName(city).toLowerCase().replace(/[^a-z0-9-]/g, "-");
  const cleanStyle = style === "modernist" ? "modernist" : "poster";
  const cleanLevel = clampLevel(level);
  // .png because flux-1.1-pro outputs PNG (schnell was WebP). Bumping the
  // version invalidates all previous .webp cache entries automatically.
  const version = CITY_PROMPT_VERSION[cleanCity] || POSTCARD_PROMPT_VERSION;
  return `${version}/${cleanCity}-L${cleanLevel}-${cleanStyle}.png`;
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
