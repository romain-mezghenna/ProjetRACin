import ollama 
from langchain_community.llms import Ollama
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
import videoEditor as ve
from typing import List
import chromadb
from chromadb.api.types import (
    Documents,
    Embeddings,
    EmbeddingFunction
)
import os

# Create the client for the chromaDB
client = chromadb.PersistentClient(path="./chromadb")

# Create the class for the embedding function
class EmbedOllama(EmbeddingFunction[Documents]):
    def __init__(self, model: str):
        self.model = model

    def __call__(self, documents: Documents) -> Embeddings:
        embeddings = []
        for document in documents:
            embedding = ollama.embeddings(model=self.model, prompt=document)
            embeddings.append(embedding.get("embedding"))
        return embeddings



# Create the collection for the script
script_collection = client.get_or_create_collection(name="script_collection", embedding_function=EmbedOllama(model="mxbai-embed-large"))

# Open the transcript file 
with open("./transcripts/video-large.txt", "r") as file:
    #Read line by line
    i=script_collection.count()
    for line in file:
        # Add the line to the collection
        script_collection.add(
            documents=[line],
            metadatas=[{"source": "video-large.txt"}],
            ids=[f"id_{i}"]
        )
        i+=1
        
print("Collection filled")

mots_alcool = [
    "bière", "vin", "whisky", "vodka", "rhum", "gin", "tequila", "liqueur",
    "champagne", "vermouth", "absinthe", "cognac", "brandy", "porto", "saké",
    "mezcal", "martini", "aperitif", "digestif", "bourbon", "cider", "scotch",
    "mojito", "margarita", "cosmopolitan", "caipirinha", "pina colada",
    "sangria", "jägermeister", "prosecco", "chianti", "mezze", "grappa",
    "kirsch", "armagnac", "aquavit", "calvados", "pastis", "ouzo", "rum",
    "punch", "amaretto", "sambuca", "baileys", "chartreuse", "kir", "cachaça",
    "pisco", "sherry", "cointreau", "daiquiri", "lager", "ale", "stout",
    "malibu", "grog", "bénédictine", "campari", "limoncello", "téquila",
    "port", "manhattan", "mimosa", "julep", "aperol", "negroni", "paloma",
    "spritz", "breezer", "gris", "rosé", "méthode champenoise", "vinho verde",
    "vin gris", "vin de pays", "vin de table", "vin de pays", "grand cru",
    "cuvée", "millésime", "assemblage", "sommelier", "oenologie", "dégustation",
    "vigne", "raisin", "cépage", "terroir", "cave", "tonneau", "barrique",
    "bouchon", "carafe", "décanteur", "carafer", "décanter", "décantation",
    "caviste", "distillerie", "brasserie", "chais", "tonnellerie", "alcoolique",
    "ivresse", "gueule de bois", "coma éthylique", "alcoolémie", "intoxication",
    "addiction", "désintoxication", "abstinence", "sobriété", "modération",
    "apéro", "soirée", "beuverie", "cuite", "biture", "biturer", "sevrage",
    "alcoolisme", "alcoolique", "alcoolisation", "alcoolo-dépendance", "ivrogne",
    "ivrognerie", "alcoolodépendance", "alcoolodépendant", "soûl", "soûlerie",
    "alcoolisation", "ivresse", "soûlographie", "poivrot", "buveur", "buverie",
    "bituré", "dégriser", "bourré", "biturage", "ivrogner", "biturer", "sobre",
    "ivrognerie", "biture", "soûlographie", "se rincer", "saoul", "rince-cervelles",
    "bituré", "se rincer", "saoulerie", "prendre une cuite", "se bourrer",
    "ivre", "soulographie", "ivre mort", "biturer", "s'éclater", "soûlographique",
    "rincé", "saouler", "cuite", "bituré", "ivre comme un polonais", "péter",
    "soulographique", "biturer", "soûl comme un polonais", "être défoncé",
    "soûler", "soûler", "être déchiré", "être pompette", "biturage", "rincé",
    "soûl", "ivre", "saoul", "être rond", "se rincer", "soûlard", "saouler",
    "soûl comme un cochon", "être bourré", "se soûler", "biture express",
    "ivresse des profondeurs", "soûlerie", "rincé", "alcoolémie", "cuite",
    "cervoise", "brasserie", "bistrot", "taverne", "bock", "bièraubeurre",
    "bistrotier", "brasserie", "boire", "buveur", "bière blonde", "bière brune",
    "bière blanche", "bière d'abbaye", "bière de garde", "bière sans alcool",
    "bière ambrée", "bière artisanale", "brasserie artisanale", "houblon",
    "bière forte", "bière légère", "mousse", "buvette", "haleine alcoolisée",
    "vin rouge", "vin blanc", "vin rosé", "vin mousseux", "vignoble",
    "vinification", "raisin", "vendange", "pressurage", "cuvaison", "fermentation",
    "décuvage", "élevage", "vinicole", "oenologue", "caviste", "sommelier",
    "carafer", "carafe", "verre à vin", "tire-bouchon", "bouteille de vin",
    "champagne", "flûte de champagne", "cuvée", "millésime", "champenois",
    "vigneron", "mousseux", "demi-sec", "sec", "brut", "extra-brut",
    "dégorgement", "liqueur d'expédition", "tirage", "remuage", "dégorgement",
    "champagne rosé", "assemblage", "crémant", "cava", "vinho verde", "vin gris",
    "vermouth", "martini", "ambré", "rouge", "blanc", "rosé", "vin doux",
    "demi-sec", "liquoreux", "moelleux", "vin de glace", "vin de paille",
    "vin jaune", "vin de liqueur", "vin muté", "vin paillé", "mistelle",
    "vin cuit"]

# Create the search tool for the script collection
class DocumentSearchTool(BaseTool):
    name: str = "DocumentSearchTool"
    description: str = "Search in French the document collection for the query texts and return the documents that are related with it."

    def _run(self, query_texts: List[str]) -> str:
        query = script_collection.query(
            query_texts=mots_alcool,
            n_results=1
        )
        return query.get("documents")


# Set the differents models

llava = Ollama(model="llava-llama3:8b")
llama = Ollama(model="llama3")
embeded = Ollama(model="mxbai-embed-large")

# Set the differents Agents
image_analyst = Agent(
    role= "Image analyst",
    goal="Analyse the preprocess image by an object recognition model and describe if any alcohol related objects are detected. Try to see if the people are drinking. You're response must be as short as possible. Be precise and do not make assumptions or approximations. Use simple language and avoid technical terms.",
    backstory="""
      You're an researcher and you want to quantify the alcohol related objects in a scene, you need to describe the scene and gives details about all the alchol related objects detected. 
      Only the objects that are related to alcohol should be described, and the a short description of the scene should be given.
      An example of a response could be :
      Timecode: 13 sec
      Scene Description: The image showcases a well-lit room with two people and multiple bottles containing red and white liquid.
      Alcohol-Related Objects: A wine glass, two bottles of red liquid, and one bottle of white liquid.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=llava
)

formater = Agent(
    role= "Formater",
    goal="Format the response of the Image analyst in a readable format.",
    backstory="""
      You're a formater, you need to format the response of the Image analyst in a readable format. 
      The response should be formatted as follows : 
      The timecode of the image
      Description of the scene (What is going on ?, how many persons ? etc...)
      List of all the alcohol related objects found in the image (or "No Alcohol" if its the case).
      Do not add any information, only format the response.
    """,
    verbose=True,
    allow_delegation=False,
    tools=[],
    llm=llama
)

# RAG With ChromaDB

document_reader = Agent(
    role="Script reader",
    goal="Read the content of the document and extract the information requested.",
    backstory="""
      You're a script reader, you need to read the content of the document and extract the information requested.
      The document contains the script of a movie, you need to search for the information requested in the script.
      If you cannot find the information in the script, you need to say so.
      You can use the script search tool to search the script for the answer as passing all the words associated with alcool in the query_texts (in French).
    """,
    verbose=True,
    allow_delegation=False,
    tools=[DocumentSearchTool(collection=script_collection)],
    llm=llama
)
# Get a random image from the video video.mp4 at 15 seconds

image = ve.extract_image("videomp4_out", 13) 

# the path of the image is ./images/video-15.png

task1 = Task(
    agent=image_analyst,
    description="""Analyze the following snapshot of the movie, it was taken at 13 sec : ./images/videomp4_out-13.png. If you cannot have the image say""",
    expected_output="""Short description of the Scene and the caracters, description of all the alcohol related objects in the scene. Say no alcohol if there is none. Tell the timecode of the image.""",
)

task2 = Task(
    agent=formater,
    description="Format the response of the Image analyst in a readable format. Do not add any information, only format the response. If the information is not clear enough, say not clear.",
    expected_output="""The timecode of the image
      Description of the scene (What is going on ?, how many persons ? etc...)
      List of all the alcohol related objects found in the image (or "No Alcohol" if its the case).
      Any form of Note or comment shouldn't be added at the end of the response as the response need to be extracted by python code.""",
)

task3 = Task(
    agent=document_reader,
    description="Is there alcohol in the scene ? If yes, what are the alcohol related objects ? If no, say no alcohol. You can use the script search tool to search the script for the answer as passing all the words associated with alcool in the query_texts (in French).",
    expected_output="The answer to the question extracted from the document.",
    tools=[DocumentSearchTool(collection=script_collection)],
)

crew1 = Crew(
    agents=[image_analyst, formater],
    tasks=[task1, task2],
    verbose=True,
)

crew2 = Crew(
    agents=[document_reader],
    tasks=[task3],
    verbose=True,
)

# Delete the image
#os.remove(image)

result = crew2.kickoff()

print(result)