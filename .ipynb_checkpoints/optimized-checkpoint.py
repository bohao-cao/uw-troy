import discord
import os
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, download_loader, ComposableGraph, GPTListIndex, GPTFaissIndex
from dotenv import load_dotenv
import faiss

def load_faiss_index(index_name="faiss_index.json", path="./index/faiss_index.index"):
    #path = os.path.join(path, index_name)
    return GPTFaissIndex.load_from_disk(
        index_name, 
        faiss_index_save_path=path
    )
    
    
def save_faiss_index(index, index_name="faiss_index.json", path="./index/faiss_index.index"):
    #path = os.path.join(path, index_name)
    index.save_to_disk(
        index_name,
        faiss_index_save_path=path
    )
    
    
def create_faiss_index(documents):
    # dimensions of text-ada-embedding-002
    d = 1536 
    faiss_index = faiss.IndexFlatL2(d)
    index = GPTFaissIndex.from_documents(documents, faiss_index=faiss_index)
    
    return index

load_dotenv()

intents = discord.Intents().all()

discord_token = os.getenv('DISCORD_TOKEN')
# channel ID for testing
# replace with you own channel ID
discord_channel = 1095448067802673244




#online_documentation = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True).load_data()
online_documentation = SimpleDirectoryReader(input_files=['./data/announcements.rtf'], exclude_hidden=True).load_data()
online_doc_index = create_faiss_index(online_documentation)
save_faiss_index(online_doc_index, "./index/online_doc_index.json", "./index/online_doc_index.index")


JSONReader = download_loader("JSONReader")
json_loader = JSONReader()
common_help_questions = json_loader.load_data('./data/help-questions.json')
common_help_questions_index = create_faiss_index(online_documentation)
save_faiss_index(common_help_questions_index, "./index/common_help_questions_index.json", "./index/common_help_questions_index.index")


# index_online_docs = GPTSimpleVectorIndex.from_documents(online_documentation)
# index_good_questions = GPTSimpleVectorIndex.from_documents(common_help_questions)

graph = ComposableGraph.from_indices(GPTListIndex, [common_help_questions_index, online_doc_index], index_summaries=["OO Docs", "Help Questions"])

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')


@client.event
async def on_thread_create(thread):
    print(f"thread:{thread}")
    message = await thread.fetch_message(thread.id)

    if thread.parent_id != discord_channel:
        return

    response = graph.query(message.content if message.content else thread.name)
    await thread.send(response)


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    message_content = message.content
    print(f"Received a question: {message_content}")
    response = graph.query(message_content)
    

    await message.channel.send(response) 
    
client.run(discord_token)



    
    




