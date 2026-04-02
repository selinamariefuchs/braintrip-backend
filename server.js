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

    const prompt = `
    Generate 5 medium to medium-hard travel trivia questions about ${city}.
    
    Return JSON in this format:
    [
      {
        "question": "...",
        "options": ["A", "B", "C", "D"],
        "correctAnswer": "A",
        "funFact": "..."
      }
    ]
    `;

    const response = await client.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
    });

    const text = response.choices[0].message.content;

    res.json({ trivia: text });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Failed to generate trivia" });
  }
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
