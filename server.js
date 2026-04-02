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

Your job is to generate HIGH-QUALITY, NON-REPETITIVE trivia.

-----------------------------------
DIFFICULTY RULES:

If mode is "standard":
- Medium difficulty
- Focus on well-known landmarks, food, neighborhoods, and culture
- Questions should feel familiar but still interesting

If mode is "challenge":
- Medium to HARD difficulty
- Focus on hidden gems, niche facts, local behavior, deeper cultural insights
- Avoid obvious tourist facts entirely
- Make questions slightly trickier but still fair
-----------------------------------

QUALITY RULES:
- Questions must feel like they come from a LOCAL or experienced traveler
- Avoid generic or overused trivia
- Avoid repeating common facts across different runs
- Make each question feel unique and memorable
- Slightly conversational tone (not textbook)

-----------------------------------

STRUCTURE:
Generate exactly 8 multiple-choice questions.

Each question must include:
- question
- 4 answer options
- correctAnswer (must match one option exactly)
- funFact (short, interesting, specific)
- category

-----------------------------------

CATEGORIES (use each at least once):
- "Neighborhoods"
- "Food & Drink"
- "Nightlife"
- "Culture & Landmarks"
- "Local Experience"

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
