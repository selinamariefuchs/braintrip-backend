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
STRICT QUESTION FOCUS:

- MOST questions MUST reference REAL, SPECIFIC places:
  - landmarks
  - attractions
  - neighborhoods
  - notable venues (bars, restaurants, clubs, etc.)

- Food questions are allowed ONLY if:
  - the item is iconic to the city
  - widely recognized (e.g., deep dish pizza in Chicago)

-----------------------------------
STRICT ANSWER FORMAT:

- Answers must be SHORT
- Prefer 1–3 words max
- Prefer proper nouns (place names)
- Examples:
  - "Central Park"
  - "Eiffel Tower"
  - "South Beach"

- DO NOT use full sentences as answers

-----------------------------------
STRICT DIFFICULTY RULES:

If mode is "standard":
- Medium difficulty
- Focus on well-known places and recognizable locations
- User should feel: "I might know this"

If mode is "challenge":
- Noticeably harder
- Focus on:
  - hidden gems
  - niche locations
  - local-only knowledge
- DO NOT include:
  - famous landmarks
  - obvious tourist spots
- User should feel: "I wouldn’t know this unless I’ve been there"

-----------------------------------
QUALITY RULES:

- NO generic trivia
- NO repeated or predictable questions
- Questions must feel engaging and slightly challenging
- Each question should feel unique and specific
- Avoid textbook or robotic phring
- Keep tone clean and consistent

-----------------------------------
STRUCTURE:

Generate exactly 8 multiple-choice questions.

Each question must include:
- question (clear and engaging)
- 4 answer options (short + clean)
- correctAnswer (must exactly match one option)
- funFact (1 short, interesting sentence)
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

Challenge mode questions MUST NOT overlap with standard-level topics.

-----------------------------------
RETURN FORMAT:

Return ONLY valid JSON.
`;

    const response = await client.responses.create({
      model: "gpt-4.1-mini",
      input: [
        {
          role: "system",
          content: "You are a precise JSON generator for a travel trivia app.",
        },
        {
          role: "user",
          content: prompt,
        },
      ],
      temperature: 1,
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
