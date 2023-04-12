import discord
import os
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, download_loader, ComposableGraph, GPTListIndex
from dotenv import load_dotenv

load_dotenv()
intents = discord.Intents().all()
discord_token = os.getenv('DISCORD_TOKEN')
# channel ID for testing
discord_channel = 1095448067802673244

JSONReader = download_loader("JSONReader")
json_loader = JSONReader()

online_documentation = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True).load_data()
discord_help_questions = json_loader.load_data('./data/help-questions.json')
index_online_docs = GPTSimpleVectorIndex.from_documents(online_documentation)
index_good_questions = GPTSimpleVectorIndex.from_documents(discord_help_questions)
graph = ComposableGraph.from_indices(GPTListIndex, [index_online_docs, index_good_questions], index_summaries=["OO Docs", "Help Questions"])

client = discord.Client(intents=intents)


@client.event
async def on_thread_create(thread):
    message = await thread.fetch_message(thread.id)
    if thread.parent_id != discord_channel:
        return

    response = graph.query(message.content if message.content else thread.name)
    await thread.send(response)


client.run(discord_token)
