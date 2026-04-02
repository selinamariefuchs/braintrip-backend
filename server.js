import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.get("/", (req, res) => {
  res.send("BrainTrip backend is running 🚀");
});

app.post("/trivia", async (req, res) => {
  try {
    const { city, mode } = req.body ?? {};

    if (!city || typeof city !== "string" || !city.trim()) {
      return res.status(400).json({ error: "City is required" });
    }

    const cleanCity = city.trim();
    const gameMode = mode === "challenge" ? "challenge" : "standard";

    const prompt = `
You are creating premium travel trivia for an app called BrainTrip.

City: ${cleanCity}
Mode: ${gameMode}

-----------------------------------
CORE GOAL:

This should feel like a FUN, DISCOVERY-BASED TRAVEL GAME — not a quiz.

Questions should feel like:
"Wait that's actually cool… I didn't know that"

-----------------------------------
STRICT QUESTION STYLE:

- MOST questions MUST reference REAL places:
  - specific bars
  - restaurants
  - neighborhoods
  - landmarks
  - attractions

- Questions should feel like FUN FACTS about places
- Avoid generic phrasing like "Which of the following..."

- Instead use:
  - "This iconic Miami beach is known for..."
  - "This neighborhood is where locals go for..."
  - "This hidden bar in Miami is famous for..."

-----------------------------------
ANSWER RULES:

- Answers must be SHORT (1–3 words ideally)
- Prefer proper nouns (place names)
- Avoid long phrases or sentences
- Wrong answers must be believable places

-----------------------------------
DIFFICULTY RULES:

If mode is "standard":
- Medium difficulty
- Focus on:
  - well-known hotspots
  - famous attractions
  - popular neighborhoods
- User should feel: "I've heard of this"

If mode is "challenge":
- MUST feel noticeably harder
- Focus ONLY on:
  - hidden gems
  - specific venues
  - local-only spots
  - niche attractions
- DO NOT include:
  - major tourist landmarks
  - obvious answers
- Make guessing harder:
  - options should be similar types of places
  - avoid giving away the answer easily

-----------------------------------
ANTI-GUESSING RULE:

Options should be similar in category so users can't guess easily.

Example:
Bad:
- Beach / Museum / Food / Park

Good:
- 4 neighborhoods
- 4 restaurants
- 4 nightlife spots

-----------------------------------
STRUCTURE:

Generate exactly 8 questions.

Each question must include:
- question (engaging, natural tone)
- options (4 similar-type answers)
- correctAnswer
- funFact (short, interesting, place-based)
- category

-----------------------------------
CATEGORIES (use each at least once):

- "Neighborhoods"
- "Food & Drink"
- "Nightlife"
- "Culture & Landmarks"
- "Local Experience"

-----------------------------------
CRITICAL RULE:

Challenge mode questions MUST NOT overlap with standard topics.

-----------------------------------
Return ONLY valid JSON.
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
                minItems: 8,
                maxItems: 8,
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

    const output = response.output_text;

    if (!output) {
      console.error("No output_text returned from OpenAI:", response);
      return res.status(500).json({ error: "No response from AI" });
    }

    let parsed;

    try {
      parsed = JSON.parse(output);
    } catch (error) {
      console.error("JSON parse error:", output);
      return res.status(500).json({ error: "Invalid JSON from AI" });
    }

    if (!parsed?.questions || !Array.isArray(parsed.questions)) {
      return res.status(500).json({ error: "Invalid trivia format from AI" });
    }

    const cleanedQuestions = parsed.questions.map((q) => {
      const options = Array.isArray(q.options) ? q.options : [];
      const correctAnswer =
        options.includes(q.correctAnswer) && typeof q.correctAnswer === "string"
          ? q.correctAnswer
          : options[0] || "";

      return {
        question: q.question,
        options,
        correctAnswer,
        funFact: q.funFact,
        category: q.category,
      };
    });

    return res.json({
      city: cleanCity,
      mode: gameMode,
      questions: cleanedQuestions,
    });
  } catch (error) {
    console.error("Trivia generation error:", error);
    return res.status(500).json({ error: "Failed to generate trivia" });
  }
});

const PORT = process.env.PORT || 10000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
