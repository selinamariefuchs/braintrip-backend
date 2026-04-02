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
    const { city } = req.body;

    if (!city || !city.trim()) {
      return res.status(400).json({ error: "City is required" });
    }

    const prompt = `
Generate 5 medium to medium-hard travel trivia questions about ${city}.

Return ONLY valid JSON as an array.
Do not use markdown.
Do not wrap the response in triple backticks.

Each item must look exactly like:
{
  "question": "string",
  "options": ["string", "string", "string", "string"],
  "correctAnswer": "string",
  "funFact": "string"
}

Rules:
- Make questions city-specific
- Focus on landmarks, neighborhoods, food, culture, nightlife, and travel experiences
- Exactly 4 options per question
- correctAnswer must exactly match one of the options
- No answer letters like A, B, C, or D
- No explanations outside the JSON array
`;

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });

    let text = response.choices[0].message.content || "";

    // Remove markdown code fences just in case
    text = text.replace(/```json\s*/g, "").replace(/```/g, "").trim();

    let parsed = JSON.parse(text);

    // Safety cleanup in case the model still returns A/B/C/D
    parsed = parsed.map((q) => {
      let correctedAnswer = q.correctAnswer;

      if (["A", "B", "C", "D"].includes(correctedAnswer)) {
        const indexMap = { A: 0, B: 1, C: 2, D: 3 };
        correctedAnswer = q.options[indexMap[correctedAnswer]];
      }

      return {
        question: q.question,
        options: q.options,
        correctAnswer: correctedAnswer,
        funFact: q.funFact,
      };
    });

    res.json({
      city,
      questions: parsed,
    });
  } catch (error) {
    console.error("Trivia generation error:", error);
    res.status(500).json({ error: "Failed to generate trivia" });
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
