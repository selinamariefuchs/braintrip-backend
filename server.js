import express from "express";
import cors from "cors";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

app.post("/trivia", async (req, res) => {
  try {
    const { city } = req.body;

    if (!city) {
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

Return ONLY valid JSON as an array.
Do not use markdown.
Do not wrap the response in triple backticks.

Each item must look exactly like:
{
  "question": "string",
  "options": ["string", "string", "string", "string"],
  "correctAnswer": "string",
  "funFact": "string",
  "category": "string"
}
`;

    console.log("🔥 Generating trivia for:", cleanCity);

    const response = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      },
      body: JSON.stringify({
        model: "gpt-5-3",
        messages: [
          {
            role: "system",
            content: "You are a precise JSON generator.",
          },
          {
            role: "user",
            content: prompt,
          },
        ],
        temperature: 1,
      }),
    });

    const data = await response.json();

    const content = data.choices?.[0]?.message?.content;

    if (!content) {
      console.error("❌ No content returned:", data);
      return res.status(500).json({ error: "No response from AI" });
    }

    let parsed;

    try {
      parsed = JSON.parse(content);
    } catch (err) {
      console.error("❌ JSON parse error:", content);
      return res.status(500).json({ error: "Invalid JSON from AI" });
    }

    console.log("✅ Trivia generated successfully");

    res.json({
      city: cleanCity,
      questions: parsed,
    });
  } catch (error) {
    console.error("❌ Server error:", error);
    res.status(500).json({ error: "Server error" });
  }
});

const PORT = process.env.PORT || 5000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
