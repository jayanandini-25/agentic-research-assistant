from typing import List, Dict
from openai import AsyncOpenAI
from config.settings import get_settings
from core.logger import setup_logger

settings = get_settings()
logger   = setup_logger(__name__)


class ReportWriter:
    """
    Phase 7 — Report writer.

    Takes all per-question summaries and writes a complete, structured
    research report in Markdown. The report includes:
    - Executive summary
    - One section per sub-question
    - Key findings
    - Cited sources list
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key  = settings.openai_api_key,
            base_url = settings.openai_base_url or None,
        )
        self.model = settings.openai_model
        logger.info(f"ReportWriter initialized | model={self.model}")

    async def write(self, query: str, summaries: List[Dict], tone: str = "objective") -> Dict:
        """
        Write the full research report.

        Args:
            query:     Original user query
            summaries: List of { question, summary, sources }
            tone:      "objective" | "analytical" | "critical"

        Returns:
            { report: str (markdown), sources: List[str] }
        """
        # Build the evidence block
        evidence_block = ""
        all_sources    = []

        for i, item in enumerate(summaries, 1):
            evidence_block += f"\n## Finding {i}: {item['question']}\n"
            evidence_block += f"{item['summary']}\n"
            for url in item.get("sources", []):
                if url not in all_sources:
                    all_sources.append(url)

        tone_instruction = {
            "objective":  "Write in a balanced, neutral, academic tone.",
            "analytical": "Write analytically, examining causes, effects, and implications.",
            "critical":   "Write critically, evaluating strengths, weaknesses, and controversies.",
        }.get(tone, "Write in a balanced, neutral, academic tone.")

        prompt = f"""You are an expert research writer. Using the research findings below, write a comprehensive research report on the topic.

Topic: {query}

{tone_instruction}

Research Findings:
{evidence_block}

Write the report in Markdown with this exact structure:

# [Descriptive title for the topic]

## Executive Summary
[2-3 sentences capturing the core answer to the topic]

## Introduction
[Brief background on why this topic matters, 2-3 sentences]

[One ## section per research finding — use the finding question as the section heading, rewritten as a statement. Write 2-3 paragraphs per section using only the evidence provided.]

## Key Findings
[Bullet list of the 5-7 most important insights across all findings]

## Conclusion
[Synthesis paragraph drawing together the key themes, 3-4 sentences]

## Sources
[Numbered list of all source URLs]

Rules:
- Minimum 600 words
- Use only information from the research findings above
- Do not add outside knowledge or hallucinate facts
- Every section must be substantive — no filler sentences
- Write the Sources section with numbered URLs

Report:"""

        try:
            response = await self.client.chat.completions.create(
                model    = self.model,
                messages = [{"role": "user", "content": prompt}],
                max_tokens  = 2000,
                temperature = 0.3,
            )
            report = response.choices[0].message.content.strip()
            logger.info(f"ReportWriter | generated {len(report)} chars | {len(all_sources)} sources")

        except Exception as e:
            logger.error(f"ReportWriter failed: {e}")
            report = f"# Research Report\n\nReport generation failed: {e}\n\n## Findings\n\n{evidence_block}"

        return {
            "report":  report,
            "sources": all_sources,
        }
