import json
import asyncio

from fastmcp import Client, FastMCP
from fastmcp.server import create_proxy
from openai import  AsyncOpenAI, OpenAI
from memory_config import memory

from mock_servers import deposit, credit, card, payroll, mem0_server, personal

import os
from dotenv import load_dotenv
load_dotenv()

client = AsyncOpenAI(
    base_url="https://ai.xazna.uz/llm/v1",
    api_key=os.getenv("XAZNA_API_KEY"),
)

_initialized = False
_cached_tools: list | None = None

main = FastMCP("Main")
mdl = None
_all_mcp_tools = None

async def initialize_mcp():
    global _initialized, main, mdl
    if _initialized:
        return

    # mdl = "gpt-4o-mini"
    m = await client.models.list()
    mdl = m.data[0].id
    print(f"Using model: {mdl}")

    # mdl = "Qwen3-32B"

    main.mount(deposit, namespace="deposit")
    main.mount(credit,  namespace="credit")
    main.mount(card,    namespace="card")
    main.mount(payroll, namespace="payroll")
    main.mount(mem0_server, namespace="memory")
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
        context_window = [conversation[0]] + conversation[-10:] if len(conversation) > 10 else conversation
        print(f"\nSending last {len(context_window)-1} messages to LLM", flush=True)
        response = await client.chat.completions.create(
            model=mdl,
            messages=context_window,
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
    USER_ID = "bank_user_002"
    
    print("Xotira yuklanmoqda...")
    past_memories = memory.search("oxirgi suhbatlar", user_id=USER_ID, limit=10)
    
    system_message = {
        "role": "system", 
        "content": (
            "Sen bank assistantisan. Senda uzoq muddatli xotira asboblari bor. "
            "Foydalanuvchi haqidagi muhim faktlarni 'memory_add_memory' orqali saqlab qo'y."
        )
    }

    conversation = [system_message]

    # Xotiradan yuklash
    if past_memories and "results" in past_memories:
        for item in reversed(past_memories["results"]): 
            conversation.append({"role": "assistant", "content": f"Eslatma: {item['memory']}"})

    while True: 
        user_input = await asyncio.to_thread(input, "\nSiz: ")
        if user_input.lower() == "exit":
            break 
        if not user_input.strip():
            continue
         
        conversation.append({"role": "user", "content": user_input})

        result = await message_streaming(conversation)
        
        if result:
            _, answer = result
            print(f"\nJavob: {answer}")


        if len(conversation) > 11:
            conversation = [system_message] + conversation[-10:]

if __name__ == "__main__":
    asyncio.run(main_loop())