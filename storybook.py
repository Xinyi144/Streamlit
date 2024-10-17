import os
import streamlit as st
from openai import OpenAI

client = OpenAI(api_key = os.environ['OPENAI_API_KEY'])

# Story
def story_gen(prompt):
  system_prompt = """
  You are a world renowned author for young author fiction ahort stories.
  Given a concept, generate a short story relevant to the themes of the concept
  The total length of the story should be within 100 words
  """

  response = client.chat.completions.create(
      model='gpt-4o-mini',
      messages=[{
          'role': 'system',
          'content': system_prompt
      }, {
          'role': 'user',
          'content': 'prompt'
      }],
      temperature=0.7,  # more higher, make it more random
      max_tokens=2000)
  return response.choices[0].message.content


# Cover art
def art_gen(prompt):
  response = client.images.generate(model='dall-e-2',
                                    prompt=prompt,
                                    size='256x256',
                                    n=1)
  return response.data[0].url


# Cover prompt design- help us design a prompt for cover prompt based on the story generated
def design_gen(prompt):
  system_prompt = """
  You will be given a short story. Generate a prompt for a cover art     that suitable for the story.
  The prompt is for dall-e-2.
  """

  response = client.chat.completions.create(model='gpt-4o-mini',
                                            messages=[{
                                                'role': 'system',
                                                'content': system_prompt
                                            }, {
                                                'role': 'user',
                                                'content': prompt
                                            }],
                                            temperature=0.7,
                                            max_tokens=1000)
  return response.choices[0].message.content


prompt = st.text_input("Enter a prompt")
if st.button("Generate"):
  story = story_gen(prompt)
  design = design_gen(story)
  art = art_gen(design)
  
  st.caption(design)
  st.divider()
  st.write(story)
  st.divider()
  st.image(art)
