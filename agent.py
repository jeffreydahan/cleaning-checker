from google.adk.agents import Agent
from google import genai
from google.genai import types

# Define a function to analyze the media and determine if cleaning is needed
async def check_if_dirty(file_uri: str) -> str:
  """Analyzes a video to determine if the floor is dirty.

  Args:
    file_uri: The Google Cloud Storage URI of the video file.

  Returns:
    A string indicating whether the room is "dirty" or "clean".
  """
  client = genai.Client(
      vertexai=True,
      project="adk-testing-458009",
      location="us-central1",
  )

  msg1_video1 = types.Part.from_uri(
      file_uri=file_uri,
      mime_type="video/quicktime",
  )

  model = "gemini-2.0-flash-001"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_video1,
        types.Part.from_text(text="""if this floor is very dirty, respond back that the room is dirty.  Otherwise, say the room is clean""")
      ]
    ),
  ]
  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    response_modalities = ["TEXT"],
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
  )

  response_text = ""
  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    response_text += chunk.text

  return response_text.strip()

# root agent definition
root_agent = Agent(
    name="cleaning_checker", # ensure no spaces here
    model="gemini-2.0-flash",
    description="Agent to check videos and images to see if they are dirty or clean",
    instruction="""I am an agent that checks file and folder locations for media files
        to see if the floors shown require cleaning. I provide this information to another agent
        which controls a robot vacuum cleaner.

        When I review a media file, I will provide an overall recommendation to clean
        a room or not to clean a room. For example, if I see a floor that has dirt or many crumbs,
        I will recommend to clean it. If there are scratches on the floor or if there is minimal dust,
        I will recommend not to clean it.

        Sample responses are:
        - The kitchen is dirty, please clean it.
        - No room is dirty, please check that the vacuum is in the dock. Then send get the status.

        Other than the sample responses above, do not provide any others - meaning, either recommend a room
        to clean, or simply ensure the vacuum is docked and then get the status.

        You can analyze two rooms.  Utilize the associated file_uri with the room for the check_if_dirty function/tool:

        kitchen: gs://adk-a2a-rooms/kitchen/IMG_3848.MOV
        hallway: gs://adk-a2a-rooms/hallway/IMG_3851.MOV
        """,
    tools=[check_if_dirty],
)