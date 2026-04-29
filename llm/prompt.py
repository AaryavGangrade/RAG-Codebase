import google.generativeai as genai
from config import Config
from typing import List, Dict

class CodebaseAgent:
    def __init__(self, search_tool):
        genai.configure(api_key=Config.GEMINI_API_KEY)
        self.model_name = Config.LLM_MODEL
        self.system_prompt = """You are an elite software architect and codebase intelligence agent.
You have access to a tool called `search_codebase` that allows you to search the codebase.
When a user asks a question, you should FIRST use the `search_codebase` tool to find relevant code, classes, and functions.
You can call the tool multiple times if you need to trace dependencies or follow execution flow.
Once you have enough context, synthesize a highly accurate, precise answer referencing specific files and code lines.
If you truly cannot find the answer after searching, state that the codebase does not contain the information.
"""
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=self.system_prompt,
            tools=[search_tool]
        )
        # We start a chat session to keep state if it needs multiple turns
        self.chat = self.model.start_chat(enable_automatic_function_calling=True)

    def generate_response(self, query: str) -> str:
        response = self.chat.send_message(query)
        return response.text
