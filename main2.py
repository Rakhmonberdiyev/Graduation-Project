import json
import asyncio

from fastmcp import Client, FastMCP
from fastmcp.server import create_proxy
from openai import  AsyncOpenAI, OpenAI
from memory_config import memory
import time

from mock_servers import deposit, credit, card, payroll,  personal

import os
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

_initialized = False
_cached_tools: list | None = None

main = FastMCP("Main")
mdl = None
_all_mcp_tools = None

async def initialize_mcp():
    global _initialized, main, mdl
    if _initialized:
        return

    mdl = "gpt-4o-mini"
    # m = await client.models.list()
    # mdl = m.data[0].id
    # print(f"Using model: {mdl}")
    # mdl = "Qwen3-32B"

    main.mount(deposit, namespace="deposit")
    main.mount(credit,  namespace="credit")
    main.mount(card,    namespace="card")
    main.mount(payroll, namespace="payroll")
    main.mount(personal, namespace="personal")
    _initialized = True


async def get_tools(namespaces: list[str] | None = None) -> list:
    global _all_mcp_tools

    if _all_mcp_tools is None:
        async with Client(main) as c:
            _all_mcp_tools = await c.list_tools()
    print(f"Total tools available: {len(_all_mcp_tools)}: {[tool.name for tool in _all_mcp_tools]}")
    selected_tools = _all_mcp_tools

    if namespaces:
        selected_tools = [
            tool for tool in _all_mcp_tools
            if any(tool.name.startswith(f"{ns}_") for ns in namespaces)
        ]
    print(f"Tools after filtering by namespaces {namespaces}: {len(selected_tools)}")

    return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                }
            }
            for tool in selected_tools
        ]

async def get_search_query(user_input: str, history: list) -> str:
    """
    Uses a fast LLM call to rewrite the user's input into a standalone 
    search query by resolving pronouns (he, it, u, bu) based on history.
    """
    if not history:
        return user_input

    # Prepare a small snippet of context for the rewriter
    context_str = ""
    for msg in history[-5:]: # Last 5 messages are enough for pronouns
        role = "Foydalanuvchi" if msg["role"] == "user" else "Assistant"
        context_str += f"{role}: {msg['content']}\n"

    prompt = f"""
Suhbat tarixidan foydalanib, foydalanuvchining oxirgi gapini mustaqil qidiruv so'roviga aylantiring.
Maqsad: 'u', 'bu', 'o'sha' kabi olmoshlarni tegishli ism yoki predmetlar bilan almashtirish.

Tarix:
{context_str}

Foydalanuvchi gapi: "{user_input}"

Faqatgina qayta yozilgan qidiruv so'rovini qaytaring (izohsiz).
"""
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Sen qidiruv so'rovlarini optimallashtiruvchi yordamchisan."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else user_input
    except Exception as e:
        print(f"Query rewrite error: {e}")
        return user_input
    
async def message_streaming(conversation=[], web_search=False):
    global main, mdl

    if web_search:
        mcp_tools = await get_tools(["real_time"])
    else:
        mcp_tools = await get_tools()

    print(len(mcp_tools), flush=True)
    new_conversation = [] 
    print(new_conversation)

    while True:
        print(f"\nSending conversation to LLM for analysis: {conversation}, tools: {len(mcp_tools)}", flush=True)

        response = await client.chat.completions.create(
            model=mdl,
            messages=conversation,
            tools=mcp_tools
        )
        # print(f"Tokens: {response.usage}") 

        message = response.choices[0].message.model_dump()
        reasoning = message.get("reasoning_content", None)

        if not reasoning and message.get("model_extra", None):
            reasoning = message.model_extra.get("reasoning_content")

        if reasoning:
            print(f"\n[🤔 Model Reasoning]:\n{reasoning}\n{'-' * 50}", flush=True)

        conversation.append(message)
        new_conversation.append(message)

        if message["content"]: #tool chaqirilganda content bo'sh bo'ladi
            return new_conversation, message["content"]

        if not message.get("tool_calls"):
            break

        for tool_call in message["tool_calls"]:
            tool_name = tool_call["function"]["name"]

            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            print(f"\n[Tool Call] {tool_name}({tool_args})", flush=True)

            try:
                result = await main.call_tool(tool_name, tool_args)

                content = ""
                for item in result.content:
                    if item.type == "text":
                        content += item.text
                    else:
                        content += f"\n[{item.type} data]"

                print(f"[Tool Result] {content[:200]}..." if len(content) > 200 else f"[Tool Result] {content}",
                      flush=True)

                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": content
                })
                new_conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": content
                })

            except Exception as e:
                print(f"Error executing tool {tool_name}: {e}")
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": f"Error executing tool: {e}"
                })
                new_conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_name,
                    "content": f"Error executing tool: {e}"
                })

        print("\nLLM is analyzing tool results (and may call more tools)...", flush=True)



async def main_loop():
    await initialize_mcp()
    print("Bank Assistant (chiqish: 'exit')")

    USER_ID = "bank_user_005"
    short_term_history: list = []  # last N user/assistant message pairs
    print(short_term_history)

    while True:
        
        user_input = await asyncio.to_thread(input, "\nSiz: ")
        start_time1 = time.time()
        if user_input.lower() == "exit":
            break

        if not user_input.strip():
           continue 
        is_simple = user_input.lower().strip() in [ #strip bila bo'sh joylarni olib tashlaymiz
        "salom", "yaxshimi", "o"
        "k", "rahmat", "ha", "yo'q", "xayr"
        ]
        memory_text = ""
        if not is_simple:
            start_time = time.perf_counter()
            search_query = await get_search_query(user_input, short_term_history)
            if search_query != user_input:
                print(f"\033[94m[Optimallashgan so'rov]: {search_query}\033[0m")
            print(f"\033[92m[Query Rewrite Time]: {(time.perf_counter() - start_time) * 1000:.2f} ms\033[0m")

            limit = 5 if len(search_query.split()) > 5 else 3
            print("Searching memory for relevant context...")
            start_time = time.perf_counter()
            memories = memory.search(search_query, user_id=USER_ID, limit=limit)
            if memories["results"]:
                memory_text = "\n".join([f"- {r['memory']}" for r in memories["results"]])
                print(f"\n[Xotira]: {memory_text}")
            end_time = time.perf_counter()
            print(f"\033[92m[Memory Search Time]: {(end_time - start_time) * 1000:.2f} ms\033[0m\n")

        # Conversation yaratiladi
        system_content = "Sen bank assistantisan."
        if memory_text:
            system_content += f"\n\nFoydalanuvchi haqida ma'lum:\n{memory_text}"

        recent_history = short_term_history[-20:]
        print(f"Recent conversation history (for context): {recent_history}")

        conversation = (
            [{"role": "system", "content": system_content}]
            + recent_history
            + [{"role": "user", "content": user_input}]
        )
        start_time2 = time.time()
        result = await message_streaming(conversation)
        print(f"\033[92m[LLM Response Time]: {(time.time() - start_time2)} ms\033[0m")

        if result:
            _, answer = result
            short_term_history.append({"role": "user", "content": user_input})
            short_term_history.append({"role": "assistant", "content": answer})
            print(f"\nJavob: {answer}")
            end_time1 = time.time()
            print(f"\033[92m[Total Response Time]: {(end_time1 - start_time1)} ms\033[0m")
            print("\033[90m(System: Saving to long-term memory in background...)\033[0m")
            # Javob Mem0g ga saqlanadi
            # asyncio.create_task(
            #     asyncio.to_thread(
            #         memory.add, 
            #         f"User: {user_input}\nAssistant: {answer}", 
            #         user_id=USER_ID
            #     )
            # )
            def save_memory_task(text, uid):
                try: 
                    start = time.perf_counter()
                    memory.add(text, user_id=uid)
                    end = time.perf_counter()
                # This will print whenever the 5 seconds are up, 
                # even while you are typing!
                    print(f"\n\033[92m[Background Success]: Memory saved in {(end - start) * 1000:.2f} ms\033[0m")
                except Exception as e:
                    print(f"\n\033[91m[Background Error]: Failed to save memory: {e}\033[0m")

            # Fire and forget
            asyncio.create_task(
                asyncio.to_thread(save_memory_task, f"User: {user_input}\nAssistant: {answer}", USER_ID)
            )

if __name__ == "__main__":
    asyncio.run(main_loop())