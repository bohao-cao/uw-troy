import fire
import discord
import os
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, download_loader, ComposableGraph, GPTListIndex, GPTTreeIndex, LLMPredictor, ServiceContext
from dotenv import load_dotenv
from llama_index.evaluation import ResponseEvaluator
from langchain.llms import OpenAI

def index_io(action="save", path="./index/online_doc", index=None):
    if action=="save":
        index.save_to_disk(path)
    else:
        return GPTTreeIndex.load_from_disk(path)
        

def create_online_doc_index(test=True):
    if test:
        # just load one file for testing
        online_documentation = SimpleDirectoryReader(input_files=['./data/announcements.rtf'], exclude_hidden=True).load_data()
    else:
        online_documentation = SimpleDirectoryReader('./data', recursive=True, exclude_hidden=True).load_data()
    return GPTTreeIndex.from_documents(online_documentation)
        

def create_faq_index():
    JSONReader = download_loader("JSONReader")
    json_loader = JSONReader()
    faq_docs = json_loader.load_data('./data/help-questions.json')
    
    return GPTTreeIndex.from_documents(faq_docs)


def init_reponse_evaluator():
    llm_predictor_gpt3 = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))
    service_context_gpt3 = ServiceContext.from_defaults(llm_predictor=llm_predictor_gpt3)

    evaluator = ResponseEvaluator(service_context=service_context_gpt3)
    return evaluator

def start(test=True, create_index=False, evaluate_response=True):   
    load_dotenv()

    # channel ID for testing
    # replace with you own channel ID
    discord_channel = 1095448067802673244

    if create_index:
        online_doc_index = create_online_doc_index()
        faq_index = create_faq_index()
        index_io("save", path="./index/online_doc", index=online_doc_index)
        index_io("save", path="./index/faq", index=faq_index)
    else:
        online_doc_index = index_io("load", path="./index/online_doc")
        faq_index = index_io("load", path="./index/faq")


    evaluator = init_reponse_evaluator() if evaluate_response else None

    graph = ComposableGraph.from_indices(GPTListIndex, [online_doc_index, faq_index], index_summaries=["OO Docs", "Help Questions"])

    intents = discord.Intents().all()
    intents.message_content = True
    
    client = discord.Client(intents=intents)
    
    @client.event
    async def on_ready():
        print(f'Logged in as user {client.user}')

    @client.event    
    async def on_thread_create(thread):
        print(f"thread:{thread}")
        message = await thread.fetch_message(thread.id)

        if thread.parent_id != discord_channel:
            return

        response = graph.query(message.content if message.content else thread.name)
        
        if evaluator:
            eval_result = evaluator.evaluate(response)
            if "YES" in eval_result:
                print("response is evaluated and positive")
            else:
                print("response is evaluated and negative, use with caution!")

        await thread.send(response)

    @client.event
    async def on_message(message):
        if message.author == client.user:
            return
        message_content = message.content
        print(f"Received a question: {message_content}")
        response = graph.query(message_content)

        if evaluator:
            eval_result = evaluator.evaluate(response)
            if "YES" in eval_result:
                print("response is evaluated and positive")
            else:
                print("response is evaluated and negative, use with caution!")
        await message.reply(response, mention_author=True)
    
    discord_token = os.getenv('DISCORD_TOKEN')
    client.run(discord_token)


if __name__ == "__main__":
    fire.Fire(start)