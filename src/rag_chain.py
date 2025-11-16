import os
import openai
from typing import List


class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline.

    Combines retrieved contexts with a query to produce LLM answers.
    """

    def __init__(self, openai_api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize RAG pipeline with OpenAI API key and model name.

        Args:
            openai_api_key: OpenAI API key (optional, can read from environment)
            model: OpenAI Chat model name
        """
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not set, please provide it or set environment variable")
        openai.api_key = openai_api_key
        self.model = model

    def build_prompt(self, query: str, contexts: List[str], max_context_chars: int = 1500) -> str:
        """
        Construct a prompt with retrieved contexts for LLM.

        Args:
            query: user question
            contexts: list of retrieved text chunks
            max_context_chars: truncate if contexts exceed this length

        Returns:
            str: full prompt to feed into LLM
        """
        # Join contexts with separator
        joined = "\n\n---\n\n".join(contexts)
        if len(joined) > max_context_chars:
            joined = joined[:max_context_chars]

        # Build prompt text
        prompt = (
                "You are an assistant who answers strictly based on the provided context.\n"
                "If the answer is not in the context, respond: 'Cannot answer based on provided knowledge.'\n\n"
                "Context:\n" + joined + "\n\nUser Question:\n" + query + "\n\nAnswer:"
        )
        return prompt

    def answer(self, query: str, contexts: List[str]) -> str:
        """
        Generate answer from LLM based on retrieved contexts.

        Args:
            query: user question
            contexts: retrieved text chunks

        Returns:
            str: LLM answer
        """
        prompt = self.build_prompt(query, contexts)
        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an assistant who strictly answers based on the context. Do not hallucinate."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.0,
        )

        if "choices" in resp and len(resp["choices"]) > 0:
            return resp["choices"][0]["message"]["content"].strip()
        return "(Failed to get a response
