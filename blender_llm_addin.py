# title: Blender AI LLM Addin for Text to 3D Graphics Model
# version: 1.0.0
# date: 2025-1-26
# authors: Taewook Kang
# email: laputa99999@gmail.com
# description: this is a Blender AI LLM Addin for generating Blender Python code using AI models like ChatGPT, Gemma, Llama, etc.
import sys, os, json, argparse, re, textwrap, ast, subprocess, sys, random, math
import bpy, pandas as pd, numpy as np

# Configuration
MAX_RETRIES = 8  # Maximum attempts to execute/fix generated code

# Create Blender UI
class OBJECT_PT_CustomPanel(bpy.types.Panel):
	bl_label = "AI Model Selector"
	bl_idname = "OBJECT_PT_custom_panel"
	bl_space_type = 'VIEW_3D'
	bl_region_type = 'UI'
	bl_category = "Gen AI 3D Graphics Model"

	def draw(self, context):
		layout = self.layout
		layout.label(text="Select Model:")
		layout.prop(context.scene, "ai_model", text="")
		# openai API key is needed for ChatGPT or custom endpoints
		if context.scene.ai_model in {'chatgpt', 'custom'}:
			layout.prop(context.scene, "openai_api_key", text="OpenAI API Key")
		# optional custom LLM endpoint and model
		if context.scene.ai_model == 'custom':
			layout.label(text="Custom LLM (OpenAI spec):")
			layout.prop(context.scene, "llm_server_url", text="Server URL")
			layout.prop(context.scene, "llm_model_name", text="Model Name")
		layout.label(text="User Prompt:")
		layout.prop(context.scene, "user_prompt", text="")
		
		layout.operator("object.submit_prompt", text="Submit")
		
		# display log messages
		if context.scene.log_messages:
			layout.separator()
			layout.label(text="Status:")
			for line in context.scene.log_messages.split('\n'):
				if line.strip():
					layout.label(text=line)

# Blender event operator
class OBJECT_OT_SubmitPrompt(bpy.types.Operator):
	bl_label = "Submit Prompt"
	bl_idname = "object.submit_prompt"

	def execute(self, context):
		option = context.scene.ai_model
		user_prompt = context.scene.user_prompt
		# pass along custom endpoint/model if provided
		server = context.scene.llm_server_url
		model = context.scene.llm_model_name
		# validate custom selection
		if option == 'custom':
			if not server or not model:
				self.report({'ERROR'}, "Custom LLM selected but Server URL and/or Model Name not provided")
				return {'CANCELLED'}
		# clear log for this run
		context.scene.log_messages = ""
		# Force UI redraw to clear previous status display
		for area in context.screen.areas:
			if area.type == 'VIEW_3D':
				area.tag_redraw()
		gen_code(option,
			'coding Blender python program using bpy, basic grammar without Explanation, "#" inline comments, complicated grammar like lamda and function under user request. do not delete the previous objects. user request is',
			user_prompt,
			server,
			model,
			context)

		return {'FINISHED'}

# Blender widget registration
def register():
	bpy.utils.register_class(OBJECT_PT_CustomPanel)
	bpy.utils.register_class(OBJECT_OT_SubmitPrompt)

	bpy.types.Scene.ai_model = bpy.props.EnumProperty(
		name="AI Model",
		items=[
				('chatgpt', "ChatGPT", ""),
				('gemma2', "Gemma", ""),
				('llama3.2', "llama", ""),
				('codellama', "codellama", ""),
				('qwen2.5-coder:3b', "qwen2.5", ""),			   
				('vanilj/Phi-4', "Phi", ""),
				('custom', "Custom LLM", ""),
			]
	)
	bpy.types.Scene.user_prompt = bpy.props.StringProperty(name="User Prompt")	# optional custom LLM endpoint (OpenAI-compatible) and model name
	bpy.types.Scene.llm_server_url = bpy.props.StringProperty(
		name="LLM Server URL",
		description="URL of a custom LLM service implementing OpenAI API spec",
		default=""
	)
	bpy.types.Scene.llm_model_name = bpy.props.StringProperty(
		name="LLM Model",
		description="Model name on the custom LLM service (e.g. llama-3.2-3b-instruct)",
		default=""
	)
	bpy.types.Scene.openai_api_key = bpy.props.StringProperty(
		name="OpenAI API Key",
		description="API key for OpenAI or OpenAI-compatible services",
		default=""
	)
	bpy.types.Scene.log_messages = bpy.props.StringProperty(
		name="Log Messages",
		description="Log output from LLM tool execution",
		default=""
	)
def unregister():
	bpy.utils.unregister_class(OBJECT_PT_CustomPanel)
	bpy.utils.unregister_class(OBJECT_OT_SubmitPrompt)

	del bpy.types.Scene.ai_model
	del bpy.types.Scene.user_prompt
	# custom LLM props
	del bpy.types.Scene.llm_server_url
	del bpy.types.Scene.llm_model_name
	del bpy.types.Scene.openai_api_key
	del bpy.types.Scene.log_messages

# Import AI, LLM libraries
import openai
from ollama import chat
from ollama import ChatResponse
from openai import OpenAI

# OpenAI model
# default OpenAI client; used when no custom endpoint is supplied
client = OpenAI(api_key='<input your OpenAI API key>')

def openai_agent(prompt, system_prompt="You are coder for Blender python program", model="gpt-4o", api_base=None, api_key=None, context=None):
	"""Send a chat completion request using OpenAI API.

	`api_key` may be supplied directly; if omitted we check
	`context.scene.openai_api_key` when a context is available.  If no key is
	found we fall back to the global `client` (which may be left empty).

	If `api_base` is provided we create a temporary client pointed at
	that base URL (useful for LM Studio or other OpenAI-spec servers).
	The openai SDK v1.x uses 'base_url' and 'api_key' in the constructor.
	"""
	# choose key from argument or context; strip whitespace that may sneak in via paste
	if api_key is None and context is not None:
		api_key = (getattr(context.scene, 'openai_api_key', None) or '').strip() or None

	# build client
	if api_base or api_key:
		base = None
		if api_base:
			base = api_base.rstrip('/')
			if not base.endswith('/v1'):
				base = base + '/v1'
		cli = OpenAI(api_key=api_key or 'lm-studio', base_url=base)
	else:
		cli = client

	try:
		response = cli.chat.completions.create(
			model=model,
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": prompt},
			],
			temperature=0.1,
			max_tokens=1024,
			top_p=1
		)
		if response is None or not response.choices:
			raise Exception(f"Invalid response from {api_base or 'OpenAI API'}: {response}")
		return response.choices[0].message.content
	except Exception as e:
		raise Exception(f"API call failed (url={api_base}, model={model}): {str(e)}")

# code agent function
def check_safe_eval(express):
	tokens = express.split()

	try:
		if tokens.index("import") < 0:
			return 
		unsafe_libs = ["os", "shutil", "subprocess", "ctypes", "pickle", "http", "socket", "eval", "exec"]
		unsafe = False
		for lib in unsafe_libs:
			try:
				if tokens.index(lib) >= 0:
					unsafe = True
					break
			except ValueError:
				pass

		if unsafe:
			raise Exception(f"{express} is not safe.")
	except ValueError:
		pass
	return 

def preprocess_code(text: str) -> str:
	try:
		match = re.search(r'```python\n(.*?)```', text, re.DOTALL) # extract code from text between ```python\n and ```
		code = match.group(1).strip()
		code = code.replace('\t', '    ')
		code = textwrap.dedent(code)
		check_safe_eval(code)
		ast.parse(code)
	except IndentationError as e:
		print(f"IndentationError detected: {e}")
		code = ''
	except SyntaxError as e:
		print(f"SyntaxError detected: {e}")
		code = ''
	except Exception as e:
		print(f"Error: {e}")
		code = ''
	return code

def llm_agent(option, prompt):
    response = chat(
        model=option,
        messages=[{"role": "user", "content": prompt}]
    )
    if response and isinstance(response, ChatResponse):
        return response.message.content
    else:
        raise Exception("Failed to get a valid response from the model")

def gen_code(option, instruct_cmd, user_prompt, server_url=None, custom_model=None, context=None):
	"""Generate and execute Blender code from LLM response."""
	def log(msg):
		"""Append message to log and print to console."""
		print(msg)
		if context and hasattr(context.scene, 'log_messages'):
			current = context.scene.log_messages
			context.scene.log_messages = (current + '\n' + msg if current else msg)[-500:]  # keep last 500 chars

	log(f"Selected Option: {option}")
	log(f"User Prompt: {user_prompt}")

	# route to appropriate LLM based on selected model
	if option == 'custom':
		log(f"Calling custom LLM at {server_url} model {custom_model}...")
		output = openai_agent(instruct_cmd + ': ' + user_prompt,
			model=custom_model,
			api_base=server_url,
			context=context)
	elif option == 'chatgpt':
		log(f'Calling {option} API...')
		output = openai_agent(instruct_cmd + ': ' + user_prompt, context=context)
	else:
		log(f'Calling {option} (ollama) API...')
		output = llm_agent(option, instruct_cmd + ': ' + user_prompt)
	log("Response received. Processing...")

	for i in range(MAX_RETRIES):
		try:
			code = preprocess_code(output)
			if not code:
				raise Exception("No valid code extracted from response")
			log(f"Executing code (attempt {i+1}/{MAX_RETRIES})...")
			exec(code)
			log("Code executed successfully.")
			break
		except Exception as e:
			log(f"Error: {str(e)[:80]}")
			if i < MAX_RETRIES - 1:
				error_fix_prompt = f"Fix the error {e} in {code}"
				if option == 'custom':
					output = openai_agent(error_fix_prompt, model=custom_model, api_base=server_url, context=context)
				elif option == 'chatgpt':
					output = openai_agent(error_fix_prompt, context=context)
				else:
					output = llm_agent(option, error_fix_prompt)

register()
