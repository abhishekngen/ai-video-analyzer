from openai import OpenAI
import json
import os
from dotenv import load_dotenv

from clients.openai_client import OpenAIClient
from services.agent_youtube_service import Youtube_Service

load_dotenv()
client = OpenAI()
openai_client = OpenAIClient()

youtube_service = Youtube_Service()

# Tool definitions

tool_functions = {
    "get_video_captions": youtube_service.get_video_captions,
    "search_video": youtube_service.search_video,
    "download_and_index_video": youtube_service.download_and_index_video,
    "index_video": youtube_service.index_video,
    "get_video_id": youtube_service.get_video_id,
    "query_embedding": youtube_service.chroma.query_embedding,
    # "label_image": openai_client.label_image,
    "get_video_frames": youtube_service.get_video_frames,
    "get_collections": youtube_service.chroma.get_collections,
}

tools = [
    {
        "type": "function",
        "name": "get_video_captions",
        "description": "Get the captions of a YouTube video. Will return the English captions if available.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "YouTube video URL"
                }
            },
            "required": ["url"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "search_video",
        "description": "Search for YouTube videos based on a query. This will return a list of video titles and URLs. You should show these to the user and ask them which video they want to download and index. Note that the search query is not guaranteed to return relevant results, so you may need to clarify with the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_query": {
                    "type": "string",
                    "description": "Search query for YouTube videos"
                }
            },
            "required": ["search_query"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "download_and_index_video",
        "description": "Download a YouTube video given its URL, and index its frames. What this means is that it will download the Youtube video from the URL, and extract its frames into a vector Chroma DB, which can be queried. The collection in the Chroma DB will be named after the video ID. There will be 2 frames indexed per second.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "YouTube video URL"
                }
            },
            "required": ["url"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_video_id",
        "description": "Get the ID of a YouTube video from its URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "YouTube video URL"
                }
            },
            "required": ["url"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "query_embedding",
        "description": "Query the Chroma Vector DB collection (named after the Video ID) using a search query. Will return n_results of the frames in the video (as images) and their timestamps whose embeddings using OpenCLIP have the closest cosine similarity to the search query. Note of course not all of these may be matching the user's query. Note that these embeddings do not know particular names of people, etc. Note the results may not include the images the user wants so you have to judge for yourself. Feel free to call multiple times to get a feel for the video and obtain results relevant to what the user wants. NOTE that the search query is used for vector cosine similarity search, thus it must be a text query relevant to the visual content of the video.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "Name of the collection"
                },
                "search_query": {
                    "type": "string",
                    "description": "Text query to search for in the video, will return the most relevant frames using RAG search. This text input will be the exact text searched up against with a cosine similarity search in the vector DB, if unhappy with results you may want to do another search since the results are not guaranteed to be relevant to the user's query. DO NOT search using particular names of people, the embedding won't know this."
                },
                "n_results": {
                    "type": ["integer", "null"],
                    "description": "(Optional) Number of top results to return, default is 8."
                },
            },
            "required": ["video_id", "search_query", "n_results"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_video_frames",
        "description": "Get two frames from a video at a specific timestamp in seconds. This is useful for getting the frames the user is referring to in their query. Note that by obtaining frames either side of a particular timestamp, you can make out the action that is happening within the video at that interval.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "Video ID (collection name in Chroma DB)"
                },
                "seconds_timestamp": {
                    "type": "integer",
                    "description": "Timestamp in seconds to get frames from. Note that two frames are sampled per second hence this function will return the two frames at that timestamp."
                }
            },
            "required": ["video_id", "seconds_timestamp"],
            "additionalProperties": False
        },
        "strict": True,
    },
    # {
    #     "type": "function",
    #     "name": "label_image",
    #     "description": "Label an image using OpenAI's image labeling function. Provide the path to the image frame you wish to label. This is very useful for actually providing information about certain parts of the video to the user. This is an expensive function to call, so use it sparingly.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "image_path": {
    #                 "type": "string",
    #                 "description": "File path to the frame you wish to label."
    #             },
    #             "label_prompt": {
    #                 "type": ["string", "null"],
    #                 "description": "(Optional) Prompt to use for labeling the image. If not provided, a default prompt will be used."
    #             }
    #         },
    #         "required": ["image_path", "label_prompt"],
    #         "additionalProperties": False
    #     },
    #     "strict": True,
    # }
    {
        "type": "function",
        "name": "get_collections",
        "description": "Get a list of all indexed video collections in the Chroma DB. This will return the names of the collections which are the video IDs.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "index_video",
        "description": "Index a video that has already been downloaded. This will extract frames from the video and index them in the Chroma DB collection named after the video ID. The frames will be indexed at 2 frames per second. YOU HAVE TO DO THIS BEFORE YOU CAN QUERY THE COLLECTION. You can check if a collection exists with your get_collections tool or simply ask the user if it has already been indexed.",
        "parameters": {
            "type": "object",
            "properties": {
                "downloaded_video_id": {
                    "type": "string",
                    "description": "Video ID (which will become the collection name in Chroma DB) of the video that has already been downloaded."
                }
            },
            "required": ["downloaded_video_id"],
            "additionalProperties": False
        },
    }
]

# Chat state
previous_response_id = None

print("Start chatting (Ctrl+C to exit):")
while True:
    inputs = []
    if not previous_response_id:
        inputs.append({
            "role": "developer",
            "content": "You are predominantly a Youtube video search assistant, you can call upon your tools to search for Youtube videos, download and index them, and query the indexed frames. You can also label frames by using the label image tool to prompt OpenAI. You are also a general assistant, you can answer questions and have conversations. You can call upon your tools to help you with this. When unsure feel free to clarify with the user. If the user is interested in a specific timestamp, you can just query the frame they are referring to and label it - but note that 2 frames are sampled per second. Do not hallucinate, verify information with your tools. Remember you have the ability to look at frames adjacent to a frame of interest to work out what is happening in the video. Use your tools freely to answer the user satisfactorily."
        })

    user_msg = input("You: ")

    inputs.append({"role": "user", "type": "message", "content": user_msg})

    # Start interaction
    response = client.responses.create(
        model="gpt-4.1",
        input=inputs,
        tools=tools,
        previous_response_id=previous_response_id,
    )

    while True:
        outputs = response.output
        tool_outputs = []

        for output in outputs:
            if output.type == "function_call":
                fn_name = output.name
                args = json.loads(output.arguments)
                print(f"[AI called function '{fn_name}' with args: {args}]")

                if fn_name in tool_functions:
                    result = tool_functions[fn_name](**args)
                else:
                    result = f"Function '{fn_name}' not implemented."

                if fn_name == "query_embedding":
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": output.call_id,
                        "output": str(result[1])
                    })

                    tool_outputs.append({
                        "role": "user",
                        "content": result[0] + [{
                            "type": "input_text", "text": "This is not actually a prompt from the user, these are the images in response to your most recent chroma DB vector search query corresponding to the above timestamps. The user cannot see this."
                        }]
                    })

                elif fn_name == "get_video_frames":
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": output.call_id,
                        "output": "The frames you requested are attached in the subsequent prompt."
                    })

                    tool_outputs.append({
                        "role": "user",
                        "content": result
                    })

                else:
                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": output.call_id,
                        "output": result
                    })

            elif output.type == "message" and output.role == "assistant":
                for item in output.content:
                    if item.type == "output_text":
                        print("AI:", item.text)

        # Exit loop if no tool outputs (i.e., the assistant has replied)
        if not tool_outputs:
            break

        # Otherwise, send the tool outputs and keep looping
        response = client.responses.create(
            model="gpt-4.1",
            input=tool_outputs,
            tools=tools,
            previous_response_id=response.id,
        )

    # Store response ID for next turn
    previous_response_id = response.id