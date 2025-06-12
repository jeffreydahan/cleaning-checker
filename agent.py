from google.adk.agents import Agent
from google import genai
from google.genai import types
import os
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Helper function to get environment variables
def get_env_var(key):
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Environment variable '{key}' not found.")
    return value

# Function to select the most recent file in a storage bucket folder
def get_most_recent_file_with_extension_check(bucket_name: str, folder: str):
  """Gets the most recent file in a GCS bucket folder and checks if its
  extension is one of .mov, .mp4, .jpg, .jpeg, .png, or .avi.

  Args:
    bucket_name: The name of the storage bucket.
    folder: The path of the folder in the storage bucket (should NOT end with a '/').

  Returns:
    A tuple containing the GCS file path of the most recent file and its mime type.

  Raises:
    ValueError: If the folder does not exist, no files are found in the folder,
                or the most recent file's extension is not one of the allowed types.
  """
  client = storage.Client()
  bucket = client.bucket(bucket_name)
  print(f"Bucket name: {bucket.name}")

  folder_prefix = folder + "/"

  # Check if the folder exists by listing blobs with the folder prefix and limiting to 1
  blobs = bucket.list_blobs(prefix=folder_prefix, max_results=1)
  if not any(blobs):
    raise ValueError(f"Folder '{folder}' does not exist in bucket '{bucket_name}'.")

  # Get all blobs within the specified folder
  blobs = bucket.list_blobs(prefix=folder_prefix)
  most_recent_blob = None

  for blob in blobs:
    if most_recent_blob is None or blob.updated > most_recent_blob.updated:
      most_recent_blob = blob

  if most_recent_blob is None:
    raise ValueError(f"No files found in folder '{folder}'.")

  _, file_extension = most_recent_blob.name.rsplit('.', 1) if '.' in most_recent_blob.name else ('', '')
  file_extension = "." + file_extension.lower()

  mime_type = None
  if file_extension == ".mov":
    mime_type = "video/quicktime"
  elif file_extension == ".mp4":
    mime_type = "video/mp4"
  elif file_extension == ".jpg":
    mime_type = "image/jpeg"
  elif file_extension == ".jpeg":
    mime_type = "image/jpeg"
  elif file_extension == ".png":
    mime_type = "image/png"
  elif file_extension == ".avi":
    mime_type = "video/x-msvideo"
  else:
    raise ValueError(f"Unrecognized file extension: '{file_extension}' for file '{most_recent_blob.name}'. "
                     f"Allowed extensions are: .mov, .mp4, .jpg, .jpeg, .png, .avi")

  return f"gs://{bucket_name}/{most_recent_blob.name}", mime_type

# Define a function to analyze the media and determine if cleaning is needed
async def check_if_dirty(room: str) -> str:
  """Analyzes a video to determine if the floor is dirty.

  Args:
    room: The name of the room to analyze

  Returns:
    A string indicating whether the room is "dirty" or "clean".
  """
  client = genai.Client(
      vertexai=True,
      project=get_env_var("GOOGLE_CLOUD_PROJECT"),
      location=get_env_var("GOOGLE_CLOUD_LOCATION"),
  )

  file, mime = get_most_recent_file_with_extension_check(get_env_var("GOOGLE_CLOUD_STORAGE_BUCKET"), room)
  msg1_video1 = types.Part.from_uri(
    file_uri = file,
    mime_type = mime,
  )

  model = "gemini-2.0-flash-001"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_video1,
        types.Part.from_text(text=
        f"""
You are analyzing media from the file: {file}
Check if the room floor in this media, which is for the room named '{room}', is dirty.
Provide your analysis in the following bulleted list format:

* Name/Path to Video File: {file}
* Floor Type: (e.g., tile, wood, carpet)
* Description: (What is visible on the floor, such as dirt, dust, toys, shoes, hair tie, etc.)
* Summary: (e.g., "The floor is very dirty," "The floor is relatively clean with minor debris," "The floor is clean.")
* Final Decision: (Based on the Summary: if the Summary indicates the floor is dirty in any way, including "relatively clean" or having any debris/items, state "The {room} is dirty, please clean it." Otherwise, if the Summary indicates the floor is clean, state "The {room} is clean, please just get the roborock status.")
""")
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
    instruction="""You are an agent that helps determine if a room's floor is dirty based on media files.
When asked to check a room (e.g., "Is the kitchen dirty?", "Check the living room floor"):
1. Identify the room name from the request.
2. Use the 'check_if_dirty' tool, passing the room name to it.
3. The 'check_if_dirty' tool will analyze the media and provide a detailed report including:
    - Name/Path to Video File
    - Floor Type
    - Description of items on the floor
    - A Summary (e.g., dirty, relatively clean, clean)
    - A Final Decision (e.g., "The [room_name] is dirty, please clean it." or "The [room_name] is clean, please just get the Roborock status.")
Your task is to call the tool and present this report to the user.
    """,
    tools=[
       check_if_dirty
    ],
)