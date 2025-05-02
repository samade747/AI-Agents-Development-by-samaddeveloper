# agents.py

from typing import Optional
from dataclasses import dataclass

from openai import AsyncOpenAI

@dataclass
class Agent:
    name: str
    instructions: str
    model: str = "gpt-4"

@dataclass
class AgentResult:
    final_output: str

class Runner:
    @staticmethod
    async def run(agent: Agent, user_input: str) -> AgentResult:
        client = AsyncOpenAI()
        system_prompt = {"role": "system", "content": agent.instructions}
        user_prompt = {"role": "user", "content": user_input}
        chat = await client.chat.completions.create(
            model=agent.model,
            messages=[system_prompt, user_prompt],
            temperature=0.7
        )
        output = chat.choices[0].message.content.strip()
        return AgentResult(final_output=output)
