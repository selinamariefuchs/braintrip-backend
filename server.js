import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import OpenAI from "openai";

dotenv.config();

if (!process.env.OPENAI_API_KEY) {
  console.error("OPENAI_API_KEY is missing. Set it in .env or env vars.");
  process.exit(1);
}

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

app.post("/itinerary", async (req, res) => {
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

const PORT = process.env.PORT || 10000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
