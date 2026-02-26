"""
🔧 Patch — The Implementer (v2.1)

Writes code. Now supports surgical Search & Replace blocks
to prevent JSON truncation on large files.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from glitchlab.agents import AgentContext, BaseAgent
from glitchlab.router import RouterResponse


# ---------------------------------------------------------------------------
# Surgical Output Schemas
# ---------------------------------------------------------------------------

class SurgicalBlock(BaseModel):
    """A single Search & Replace operation."""
    search: str = Field(..., description="The EXACT snippet of code to look for.")
    replace: str = Field(..., description="The code that should replace the search block.")

class FileChange(BaseModel):
    file: str
    action: Literal["modify", "create", "delete"]
    content: str | None = None
    surgical_blocks: list[SurgicalBlock] = Field(default_factory=list)
    description: str = "automated change" # Default value prevents validation errors

class TestChange(BaseModel):
    file: str
    content: str
    description: str = ""  # Make this optional

class ImplementationResult(BaseModel):
    changes: list[FileChange] = Field(default_factory=list)
    tests_added: list[TestChange] = Field(default_factory=list)
    commit_message: str = "chore: implementation update"  # Give a safe default
    summary: str = "Implementation completed."


# ---------------------------------------------------------------------------
# Agent Implementation
# ---------------------------------------------------------------------------

class ImplementerAgent(BaseAgent):
    role = "implementer"

    system_prompt = """You are Patch, the surgical implementation engine.

You MUST respond with a valid JSON object. No markdown wrapping.

STRATEGY FOR LARGE FILES (>100 lines):
- Do NOT rewrite the whole file in 'content'. This causes JSON truncation.
- Instead, use 'surgical_blocks' to perform Search & Replace edits.
- Each block must contain enough unique code in 'search' to be found accurately.

STRATEGY FOR SMALL FILES (<100 lines):
- Provide the FULL file in 'content' and leave 'surgical_blocks' empty.

Output schema:
{
  "changes": [
    {
      "file": "path/to/file",
      "action": "modify",
      "surgical_blocks": [
        {
          "search": "def old_func():\\n    pass",
          "replace": "def new_func():\\n    return True"
        }
      ],
      "description": "surgical update"
    }
  ],
  "tests_added": [...],
  "commit_message": "...",
  "summary": "..."
}

CRITICAL RULES:
1. Whitespace in 'search' blocks must be EXACT.
2. If using 'surgical_blocks', leave 'content' as null.
3. For NEW files (action='create'), always use 'content'.
"""

    def run(self, context: AgentContext, **kwargs) -> dict[str, Any]:
        kwargs["response_format"] = {"type": "json_object"}
        return super().run(context, **kwargs)

    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        state = context.previous_output
        
        # Determine if we should nudge the model toward surgical edits
        files_in_scope = state.get("files_in_scope", [])
        is_large_task = len(files_in_scope) > 2 or state.get("estimated_complexity") == "high"

        steps_text = ""
        for step in state.get("plan_steps", []):
            steps_text += f"\nStep {step.get('step_number')}: {step.get('description')}\n"

        file_context = ""
        if context.file_context:
            file_context = "\n\nCurrent file contents:\n"
            for fname, content in context.file_context.items():
                file_context += f"\n--- {fname} ---\n{content}\n"

        user_content = f"""Task: {context.objective}
Plan: {steps_text}
{file_context}

IMPORTANT: If modifying large files, use 'surgical_blocks' to avoid JSON truncation."""

        return [self._system_msg(), self._user_msg(user_content)]

    def parse_response(self, response: RouterResponse, context: AgentContext) -> dict[str, Any]:
        content = response.content.strip()
        
        # Phase 1: Markdown Cleaning
        content = re.sub(r"^```json\s*", "", content, flags=re.MULTILINE)
        content = re.sub(r"^```\s*", "", content, flags=re.MULTILINE)
        content = content.strip("`").strip()

        # Phase 2: Standard Parse
        try:
            raw_json = json.loads(content)
            # Fix schema quirks on the fly
            if "changes" in raw_json:
                for c in raw_json["changes"]:
                    if not c.get("description"): c["description"] = "automatic update"
            if "tests_added" in raw_json and isinstance(raw_json["tests_added"], list):
                raw_json["tests_added"] = [t for t in raw_json["tests_added"] if isinstance(t, dict)]
                
            return ImplementationResult(**raw_json).model_dump()
            
        except Exception as e:
            logger.warning(f"[IMPLEMENTER] JSON parse failed ({e}). Running Emergency Extraction...")

            # Phase 3: The Surgical Extraction
            f_match = re.search(r'"file":\s*"([^"]+)"', content)
            c_match = re.search(r'"content":\s*"(.*?)"(?=\s*[,}\n])', content, re.DOTALL)
            
            if f_match and c_match:
                filename = f_match.group(1)
                code = c_match.group(1).replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
                
                logger.info(f"[IMPLEMENTER] SURGERY SUCCESS: Extracted {filename} from broken JSON.")
                return {
                    "changes": [{
                        "file": filename,
                        "action": "create", 
                        "content": code,
                        "description": "Recovered via Emergency Extraction"
                    }],
                    "tests_added": [],
                    "commit_message": "feat: auto-creation via recovery",
                    "summary": "Recovered file content from malformed LLM response."
                }
            
            return self._fallback_result(content, str(e))

        except Exception as e:
            logger.warning(f"[IMPLEMENTER] JSON broken, attempting Emergency Extraction: {e}")

            # --- PHASE 3: THE EMERGENCY EXTRACTION ---
            # We look for the "file" and "content" values directly using regex.
            # This works even if the JSON is missing braces, commas, or has trailing text.
            file_match = re.search(r'"file":\s*"([^"]+)"', content)
            
            # This regex captures everything between the "content" quotes. 
            # It's greedy but stops at the last potential quote before the next key or end of object.
            code_match = re.search(r'"content":\s*"(.*?)"(?=\s*[,}])', content, re.DOTALL)
            
            if file_match and code_match:
                filename = file_match.group(1)
                # Unescape common JSON characters so the code is actually valid Python
                extracted_code = code_match.group(1).replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
                
                logger.info(f"[IMPLEMENTER] Successfully extracted {filename} via Emergency Regex!")
                return {
                    "changes": [{
                        "file": filename, 
                        "action": "modify", # Use modify since we'll 'touch' it first
                        "content": extracted_code, 
                        "description": "Recovered via Emergency Extraction"
                    }],
                    "tests_added": [],
                    "commit_message": "feat: audit logger (recovered)",
                    "summary": "The LLM's JSON was malformed, but the implementation logic was surgically recovered."
                }
            
            # If even the regex fails, we admit defeat
            return self._fallback_result(content, str(e))

    @staticmethod
    def _fallback_result(raw: str, error: str) -> dict[str, Any]:
        return {
            "changes": [],
            "tests_added": [],
            "commit_message": "fix: implementation (parse error)",
            "summary": f"Failed to parse: {error}",
            "parse_error": True,
            "raw_response": raw[:2000],
        }