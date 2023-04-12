import discord
import os
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, download_loader, ComposableGraph, GPTListIndex
from dotenv import load_dotenv

load_dotenv()
intents = discord.Intents().all()
discord_token = os.getenv('DISCORD_TOKEN')
discord_channel = 1083424140813410396

JSONReader = download_loader("JSONReader")
json_loader = JSONReader()

online_documentation = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True).load_data()
discord_help_questions = json_loader.load_data('./data/help-questions.json')
index1 = GPTSimpleVectorIndex.from_documents(online_documentation)
index2 = GPTSimpleVectorIndex.from_documents(discord_help_questions)
graph = ComposableGraph.from_indices(GPTListIndex, [index1, index2], index_summaries=["OO Docs", "Help Questions"])

client = discord.Client(intents=intents)


@client.event
async def on_thread_create(thread):
    message = await thread.fetch_message(thread.id)
    if thread.parent_id != discord_channel:
        return

    response = graph.query(message.content if message.content else thread.name)
    await thread.send(response)


client.run(discord_token)
