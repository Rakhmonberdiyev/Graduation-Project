import asyncio
import json
import os
import re
import time
from collections import defaultdict

import rag
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
from fastmcp import Client, FastMCP
from memory_config import memory
from mock_servers import card, credit, deposit, payroll, personal
from openai import AsyncOpenAI
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
console = Console()

user_histories: dict[int, list] = defaultdict(list)
user_active_doc: dict[int, str | None] = defaultdict(lambda: None)

_mcp = FastMCP("TgMain")
_mcp_initialized = False
_all_mcp_tools: list | None = None
MDL = "gpt-4o-mini"


# ── MCP setup ──────────────────────────────────────────────────────────────────

async def initialize_mcp():
    global _mcp_initialized
    if _mcp_initialized:
        return
    _mcp.mount(deposit,  namespace="deposit")
    _mcp.mount(credit,   namespace="credit")
    _mcp.mount(card,     namespace="card")
    _mcp.mount(payroll,  namespace="payroll")
    _mcp.mount(personal, namespace="personal")
    _mcp_initialized = True


async def get_tools() -> list:
    global _all_mcp_tools
    if _all_mcp_tools is None:
        async with Client(_mcp) as c:
            _all_mcp_tools = await c.list_tools()
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema or {"type": "object", "properties": {}},
            },
        }
        for t in _all_mcp_tools
    ]


# ── Rich logging helpers ────────────────────────────────────────────────────────

def log_incoming(tg_id: int, username: str, text: str):
    console.rule("[bold cyan] New Message [/bold cyan]")
    t = Table(show_header=False, box=None, padding=(0, 3))
    t.add_column("k", style="dim", width=14)
    t.add_column("v")
    t.add_row("Telegram ID", str(tg_id))
    t.add_row("Username", escape(username))
    t.add_row("Message", escape(text))
    console.print(Panel(t, title="[cyan]📨 Incoming[/cyan]", border_style="cyan"))


def log_polish(original: str, polished: str, ms: float):
    clean = polished.strip('"')
    changed = clean != original
    t = Table(show_header=False, box=None, padding=(0, 3))
    t.add_column("k", style="dim", width=12)
    t.add_column("v")
    t.add_row("Original", escape(original))
    t.add_row("Polished", escape(polished))
    t.add_row("Changed?", "[green]Yes ✓[/green]" if changed else "[dim]No[/dim]")
    t.add_row("Time", f"{ms:.1f} ms")
    console.print(Panel(t, title="[green]✏️  Query Polish[/green]", border_style="green"))


def log_rag_search(query: str, hits: list, all_hits: list, ms: float):
    console.print(Panel(
        f"Query: [italic]\"{escape(query)}\"[/italic]   "
        f"Returned: {len(all_hits)}   "
        f"Passed threshold: {len(hits)}   "
        f"Time: {ms:.1f} ms",
        title="[magenta]📄 RAG Search[/magenta]",
        border_style="magenta",
    ))
    if all_hits:
        t = Table(show_header=True, header_style="bold")
        t.add_column("#", width=5)
        t.add_column("Score", width=10)
        t.add_column("Passed", width=8)
        t.add_column("File", width=24)
        t.add_column("Preview")
        for i, h in enumerate(all_hits, 1):
            passed = h in hits
            color = "green" if passed else "red"
            t.add_row(
                str(i),
                f"[{color}]{h['score']}[/{color}]",
                f"[{color}]{'✓' if passed else '✗'}[/{color}]",
                escape(h["filename"]),
                escape(h["chunk"][:80]),
            )
        console.print(t)


def log_memory_search(query: str, results: list, ms: float):
    console.print(Panel(
        f"Query: [italic]\"{escape(query)}\"[/italic]   "
        f"Results: {len(results)}   "
        f"Time: {ms:.1f} ms",
        title="[blue]🔍 Mem0 Search[/blue]",
        border_style="blue",
    ))
    if results:
        t = Table(show_header=True, header_style="bold")
        t.add_column("#", width=5)
        t.add_column("Score", width=10)
        t.add_column("Memory")
        for i, r in enumerate(results, 1):
            score = r.get("score", "?")
            score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
            t.add_row(str(i), score_str, escape(r["memory"]))
        console.print(t)


def log_context_source(source: str):
    color = "magenta" if source == "RAG" else "yellow"
    console.print(Panel(
        f"[{color}]📄 Answer will be grounded in: {source}[/{color}]",
        border_style=color,
    ))


def log_prompt_sent(system: str, history_len: int, user_msg: str):
    console.print(Panel(
        f"[bold]SYSTEM[/bold]\n{escape(system)}\n\n"
        f"[bold]USER[/bold]\n{escape(user_msg)}\n\n"
        f"History turns: {history_len}",
        title="[yellow]📤 Sending to LLM[/yellow]",
        border_style="yellow",
    ))


def log_llm_answer(answer: str, ms: float, source: str):
    console.print(Panel(
        f"{escape(answer)}\n\n[dim]Response time: {ms:.1f} ms[/dim]",
        title="[green]🤖 LLM Answer[/green]",
        subtitle=f"[dim]source: {source}[/dim]",
        border_style="green",
    ))


def log_doc_upload(filename: str, mime: str, chunks: int, chars: int, ms: float):
    t = Table(show_header=False, box=None, padding=(0, 3))
    t.add_column("k", style="dim", width=10)
    t.add_column("v")
    t.add_row("File", escape(filename))
    t.add_row("Type", escape(mime))
    t.add_row("Chunks", str(chunks))
    t.add_row("Chars", str(chars))
    t.add_row("Time", f"{ms:.1f} ms")
    console.print(Panel(t, title="[cyan]📄 Document Uploaded[/cyan]", border_style="cyan"))


def log_total(ms: float):
    console.rule(f"Total: {ms:.1f} ms")


# ── LLM helpers ────────────────────────────────────────────────────────────────

async def get_search_query(user_input: str, history: list) -> tuple[str, float]:
    if not history:
        return user_input, 0.0
    context_str = ""
    for msg in history[-5:]:
        role = "Foydalanuvchi" if msg["role"] == "user" else "Assistant"
        context_str += f"{role}: {msg['content']}\n"
    prompt = (
        f"Suhbat tarixidan foydalanib, foydalanuvchining oxirgi gapini mustaqil "
        f"qidiruv so'roviga aylantiring. 'u', 'bu', 'o'sha' kabi olmoshlarni "
        f"tegishli ism yoki predmetlar bilan almashtiring.\n\n"
        f"Tarix:\n{context_str}\n"
        f"Foydalanuvchi gapi: \"{user_input}\"\n\n"
        f"Faqatgina qayta yozilgan qidiruv so'rovini qaytaring (izohsiz)."
    )
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=MDL,
            messages=[
                {"role": "system", "content": "Sen qidiruv so'rovlarini optimallashtiruvchi yordamchisan."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        rewritten = resp.choices[0].message.content.strip()
    except Exception:
        rewritten = user_input
    ms = (time.perf_counter() - t0) * 1000
    return (rewritten if rewritten else user_input), ms


async def translate_to_english(text: str) -> str:
    resp = await client.chat.completions.create(
        model=MDL,
        messages=[
            {"role": "system", "content": "Translate the user's text to English. Return only the translation, nothing else."},
            {"role": "user", "content": text},
        ],
        temperature=0,
        max_tokens=200,
    )
    return resp.choices[0].message.content.strip()


async def summarize_for_memory(user_input: str, answer: str, filename: str) -> str:
    resp = await client.chat.completions.create(
        model=MDL,
        messages=[
            {"role": "system", "content": "Summarize the following Q&A into one concise sentence for long-term memory storage."},
            {"role": "user", "content": f"Document: {filename}\nQ: {user_input}\nA: {answer}"},
        ],
        temperature=0,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


def save_memory_sync(text: str, user_id: str):
    try:
        t0 = time.perf_counter()
        memory.add(text, user_id=user_id)
        ms = (time.perf_counter() - t0) * 1000
        t = Table(show_header=False, box=None, padding=(0, 3))
        t.add_column("k", style="dim", width=12)
        t.add_column("v")
        t.add_row("User ID", user_id)
        t.add_row("Content", escape(text[:120] + ("…" if len(text) > 120 else "")))
        t.add_row("Status", "[green]✓ Saved[/green]")
        t.add_row("Time", f"{ms:.1f} ms")
        console.print(Panel(t, title="[blue]💾 Mem0 Save[/blue]", border_style="blue"))
    except Exception as e:
        console.print(f"[red]Memory save error: {e}[/red]")


async def message_streaming(conversation: list, use_tools: bool = True) -> tuple[list, str] | None:
    mcp_tools = await get_tools() if use_tools else []
    new_conversation: list = []

    while True:
        kwargs: dict = {"model": MDL, "messages": conversation}
        if mcp_tools:
            kwargs["tools"] = mcp_tools

        response = await client.chat.completions.create(**kwargs)
        message = response.choices[0].message.model_dump()
        conversation.append(message)
        new_conversation.append(message)

        if message.get("content"):
            return new_conversation, message["content"]

        if not message.get("tool_calls"):
            break

        for tool_call in message["tool_calls"]:
            tool_name = tool_call["function"]["name"]
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            console.print(f"[yellow]🔧 Tool call:[/yellow] {tool_name}({tool_args})")

            try:
                result = await _mcp.call_tool(tool_name, tool_args)
                content = "".join(
                    item.text if item.type == "text" else f"\n[{item.type} data]"
                    for item in result.content
                )
            except Exception as e:
                content = f"Error: {e}"

            console.print(f"[dim]   └─ {content[:120]}[/dim]")
            tool_msg = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": content,
            }
            conversation.append(tool_msg)
            new_conversation.append(tool_msg)

    return None


# ── Handlers ───────────────────────────────────────────────────────────────────

@dp.message(CommandStart())
async def handle_start(message: Message):
    tg_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    console.print(Panel(
        f"[cyan]/start[/cyan] from [yellow]{tg_id}[/yellow] (@{username})",
        title="[cyan]👋 Start[/cyan]", border_style="cyan",
    ))
    await message.answer(
        "👋 Assalomu alaykum! Men sizning bank assistantingizman.\n\n"
        "Quyidagi imkoniyatlardan foydalanishingiz mumkin:\n\n"
        "💬 *Bank bo'yicha savol bering*\n"
        "Istalgan bank xizmati haqida so'rang — karta, kredit, depozit va boshqalar.\n\n"
        "📄 *Hujjat yuklang*\n"
        "PDF yoki TXT fayl yuboring — men uni o'qib, shu hujjat asosida savollaringizga javob beraman.\n\n"
        "💳 *Karta raqami bo'yicha ma'lumot*\n"
        "Karta raqamingizni yozing va men u haqida ma'lumot beraman.\n"
        "Masalan: `22618000060725417701`\n\n"
        "📋 *Buyruqlar:*\n"
        "🧠 /memory — siz haqingizda saqlangan ma'lumotlarni ko'rish\n"
        "📂 /docs — yuklangan hujjatlaringiz ro'yxati\n"
        "🗑 /clear\\_docs — barcha hujjatlarni o'chirish",
        parse_mode="Markdown",
    )


@dp.message(Command("memory"))
async def handle_memory(message: Message):
    user_id = f"tg_{message.from_user.id}"
    console.print(Panel(f"[blue]/memory for {user_id}[/blue]", border_style="blue"))
    try:
        results = await asyncio.to_thread(memory.search, "user preferences and history", user_id=user_id, limit=20)
        items = results.get("results", [])
        if not items:
            await message.answer("🧠 Hozircha siz haqingizda hech narsa saqlanmagan.")
            return
        lines = "\n".join(f"{i+1}. {r['memory']}" for i, r in enumerate(items))
        await message.answer(f"🧠 *Siz haqingizda saqlangan ma'lumotlar:*\n\n{lines}", parse_mode="Markdown")
    except Exception as e:
        await message.answer(f"Xatolik: {e}")


@dp.message(Command("docs"))
async def handle_docs(message: Message):
    user_id = f"tg_{message.from_user.id}"
    docs = await asyncio.to_thread(rag.list_documents, user_id)
    if not docs:
        await message.answer("📂 Hujjatlar yuklanmagan.")
        return
    active = user_active_doc[message.from_user.id]
    lines = "\n".join(
        f"{i+1}. {d}{'  ← faol' if d == active else ''}"
        for i, d in enumerate(docs)
    )
    await message.answer(
        f"📂 *Yuklangan hujjatlar:*\n\n{lines}\n\n"
        "Faol hujjatni o'zgartirish uchun raqamini yozing (masalan: `1`)",
        parse_mode="Markdown",
    )


@dp.message(Command("clear_docs"))
async def handle_clear_docs(message: Message):
    user_id = f"tg_{message.from_user.id}"
    await asyncio.to_thread(rag.delete_documents, user_id)
    user_active_doc[message.from_user.id] = None
    console.print(Panel(f"[red]Docs cleared for {user_id}[/red]", border_style="red"))
    await message.answer("✅ Barcha hujjatlaringiz o'chirildi.")


@dp.message(F.text.regexp(r"^\d{1,3}$"))
async def handle_doc_switch(message: Message):
    tg_user_id = message.from_user.id
    user_id = f"tg_{tg_user_id}"
    docs = await asyncio.to_thread(rag.list_documents, user_id)
    if not docs:
        return
    idx = int(message.text) - 1
    if 0 <= idx < len(docs):
        user_active_doc[tg_user_id] = docs[idx]
        console.print(Panel(
            f"[green]Active doc → [bold]{docs[idx]}[/bold][/green]",
            title="[magenta]📌 Doc Switch[/magenta]", border_style="magenta",
        ))
        await message.answer(f"📌 Faol hujjat: *{docs[idx]}*", parse_mode="Markdown")


@dp.message(F.document)
async def handle_document(message: Message):
    doc = message.document
    mime = doc.mime_type or ""
    filename = doc.file_name or "document"

    if "pdf" not in mime and "text" not in mime:
        await message.answer("Faqat PDF yoki TXT fayllar qabul qilinadi.")
        return

    user_id = f"tg_{message.from_user.id}"
    await message.answer(f"📄 '{filename}' qabul qilindi, qayta ishlanmoqda…")
    await bot.send_chat_action(message.chat.id, ChatAction.UPLOAD_DOCUMENT)

    file = await bot.get_file(doc.file_id)
    buf = await bot.download_file(file.file_path)
    file_bytes = buf.read()

    t0 = time.perf_counter()
    stats = await asyncio.to_thread(rag.store_document, file_bytes, mime, filename, user_id)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    user_active_doc[message.from_user.id] = filename
    log_doc_upload(filename, mime, stats["chunks"], stats["chars"], elapsed_ms)
    console.print(Panel(
        f"[green]Active doc set → [bold]{filename}[/bold][/green]",
        title="[magenta]📌 Active Document[/magenta]", border_style="magenta",
    ))
    await message.answer(
        f"✅ '{filename}' saqlandi.\n"
        f"📊 {stats['chunks']} bo'lak, {stats['chars']} belgi.\n"
        f"Endi shu hujjat bo'yicha savol bering!",
    )


@dp.message(F.text)
async def handle_message(message: Message):
    tg_user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name or str(tg_user_id)
    user_input = message.text.strip()
    user_id = f"tg_{tg_user_id}"
    total_start = time.perf_counter()

    if not user_input:
        return

    log_incoming(tg_user_id, username, user_input)
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    short_term_history = user_histories[tg_user_id]
    is_simple = user_input.lower() in ["salom", "yaxshimi", "ok", "rahmat", "ha", "yo'q", "xayr"]
    is_id_lookup = bool(re.fullmatch(r"\d{10,}", user_input))

    # ── 1. Polish query ────────────────────────────────────────────────────────
    search_query, polish_ms = await get_search_query(user_input, short_term_history)
    if not is_simple:
        log_polish(user_input, search_query, polish_ms)

    # ── 2. RAG search ──────────────────────────────────────────────────────────
    rag_hits: list = []
    rag_all: list = []
    active_doc = user_active_doc[tg_user_id]

    has_docs = await asyncio.to_thread(rag.has_documents, user_id) if not is_simple else False
    use_rag = has_docs and not is_id_lookup

    if use_rag:
        rag_query = await translate_to_english(search_query)
        if rag_query != search_query:
            console.print(Panel(
                f"[dim]Original:[/dim] {escape(search_query)}\n"
                f"[green]Translated:[/green] {escape(rag_query)}",
                title="[magenta]🌐 RAG Query Translation[/magenta]",
                border_style="magenta",
            ))
        if active_doc:
            console.print(f"[dim]📌 Searching only active doc: [bold]{active_doc}[/bold][/dim]")
        t0 = time.perf_counter()
        rag_hits, rag_all = await asyncio.to_thread(
            rag.search_documents, rag_query, user_id, active_doc
        )
        log_rag_search(rag_query, rag_hits, rag_all, (time.perf_counter() - t0) * 1000)

    # ── 3. Mem0 search ─────────────────────────────────────────────────────────
    memory_text = ""
    if not is_simple:
        t0 = time.perf_counter()
        mem_results = await asyncio.to_thread(memory.search, search_query, user_id=user_id, limit=5)
        search_ms = (time.perf_counter() - t0) * 1000
        mem_list = mem_results.get("results", [])
        log_memory_search(search_query, mem_list, search_ms)
        if mem_list:
            memory_text = "\n".join(f"- {r['memory']}" for r in mem_list)

    # ── 4. Build system prompt ─────────────────────────────────────────────────
    log_context_source("RAG" if use_rag else "Mem0 + Tools")

    if use_rag:
        if rag_hits:
            doc_context = "\n\n".join(
                f"[{h['filename']} | score {h['score']}]\n{h['chunk']}"
                for h in rag_hits
            )
            rag_instruction = (
                "Foydalanuvchi yuklagan hujjatdan savol bermoqda. "
                "Faqat quyidagi hujjat bo'laklariga asoslanib javob ber. "
                "Agar javob hujjatda bo'lmasa, shuni aniq ayt.\n\n"
                f"[HUJJAT KONTEKSTI]\n{doc_context}\n[/HUJJAT KONTEKSTI]"
            )
        else:
            doc_names = await asyncio.to_thread(rag.list_documents, user_id)
            names_str = ", ".join(doc_names) if doc_names else "noma'lum"
            rag_instruction = (
                f"Foydalanuvchi '{names_str}' hujjatini yuklagan, "
                "lekin bu savol uchun hujjatda tegishli bo'lak topilmadi. "
                "Foydalanuvchiga hujjatda bu ma'lumot topilmaganini aniq ayt "
                "va mavzu bo'yicha boshqacha so'z bilan so'rab ko'rishini taklif qil."
            )
        system_content = f"Sen bank assistantisan.\n\n{rag_instruction}"
        if memory_text:
            system_content += f"\n\nFoydalanuvchi haqida qo'shimcha: {memory_text}"
    else:
        system_content = "Sen bank assistantisan."
        if memory_text:
            system_content += f"\n\nFoydalanuvchi haqida ma'lum:\n{memory_text}"

    # ── 5. Call LLM ────────────────────────────────────────────────────────────
    recent_history = short_term_history[-20:]
    log_prompt_sent(system_content, len(recent_history), user_input)

    conversation = (
        [{"role": "system", "content": system_content}]
        + recent_history
        + [{"role": "user", "content": user_input}]
    )

    llm_start = time.perf_counter()
    result = await message_streaming(conversation, use_tools=not use_rag)
    llm_ms = (time.perf_counter() - llm_start) * 1000

    if result:
        _, answer = result
        log_llm_answer(answer, llm_ms, source="RAG" if use_rag else "Mem0+Tools")

        user_histories[tg_user_id].append({"role": "user", "content": user_input})
        user_histories[tg_user_id].append({"role": "assistant", "content": answer})

        await message.answer(answer)

        # ── 6. Save to Mem0 ───────────────────────────────────────────────────
        if use_rag and rag_hits:
            filename_used = rag_hits[0]["filename"]
            summary = await summarize_for_memory(user_input, answer, filename_used)
            mem_text = summary
            console.print(Panel(
                f"[dim]RAG → Mem0 summary:[/dim] [italic]{escape(summary)}[/italic]",
                title="[blue]🔄 RAG → Mem0[/blue]", border_style="blue",
            ))
        else:
            mem_text = f"User: {user_input}\nAssistant: {answer}"

        asyncio.create_task(asyncio.to_thread(save_memory_sync, mem_text, user_id))
    else:
        await message.answer("Kechirasiz, javob ololmadim. Qayta urinib ko'ring.")

    log_total((time.perf_counter() - total_start) * 1000)


# ── Entry point ────────────────────────────────────────────────────────────────

async def main():
    await initialize_mcp()
    console.print(Panel(
        "[bold green]Bot polling started.[/bold green]\nWaiting for messages…",
        border_style="green",
    ))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
