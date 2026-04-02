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
    const { city } = req.body ?? {};

    if (!city || typeof city !== "string" || !city.trim()) {
      return res.status(400).json({ error: "City is required" });
    }

    const cleanCity = city.trim();

    const prompt = `
You are creating premium travel trivia for an app called BrainTrip.

Generate exactly 5 multiple-choice trivia questions about ${cleanCity}.

This is NOT generic tourist trivia.
The questions should feel like they were written by someone who actually knows the city.

The trivia should feel:
- city-specific
- modern
- slightly insider
- fun for travelers
- medium to medium-hard
- specific enough that the user learns something useful or memorable

You must diversify the 5 questions across these types:
1. neighborhood / area vibe
2. food or drink
3. nightlife or entertainment
4. landmark or cultural spot
5. local experience / hidden gem / travel behavior

Avoid:
- overly obvious tourist facts
- the most generic first-result internet trivia
- textbook phrasing
- dull history-only questions
- questions where the answer is too easy to guess immediately

Rules:
- Exactly 5 questions
- Exactly 4 answer options per question
- Wrong answers should be believable
- correctAnswer must exactly match one of the options
- funFact should be short, interesting, and specific
- category must be one of:
  "Neighborhoods",
  "Food & Drink",
  "Nightlife",
  "Culture & Landmarks",
  "Local Experience"

Return valid JSON only.
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
