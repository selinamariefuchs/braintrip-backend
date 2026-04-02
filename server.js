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
                  },
                  required: [
                    "question",
                    "options",
                    "correctAnswer",
                    "funFact",
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
          category: typeof q.category === "string" ? q.category : "Local Experience",
        };
      })
      .filter((q) => q.question && q.options.length === 4 && q.correctAnswer && q.funFact);

    const uniqueQuestions = cleanedQuestions.filter(
      (q) => !seenQuestions.includes(q.question)
    );

    let finalQuestions = uniqueQuestions;
    if (finalQuestions.length < 5) {
      // Fallback: keep unique + available, no duplicates
      finalQuestions = cleanedQuestions.slice(0, 5);
    }

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

const PORT = process.env.PORT || 10000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
