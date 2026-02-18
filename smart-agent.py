import os
import sys
import json
import subprocess
import argparse
import platform

# Suppress logging warnings from gRPC/absl
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'

# --- Configuration Management ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "agent_config.json")
MEMORY_FILE = os.path.join(BASE_DIR, "agent_memory.json")

# --- Configuration Management ---
def load_config():
    if not os.path.exists(CONFIG_FILE):
        return None
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def setup_wizard():
    print("Welcome to Smart Agent Setup!")
    current_config = load_config() or {}
    
    # Provider Selection
    print("Please select your AI Provider:")
    print("1. Google Gemini")
    print("2. OpenAI")
    print("3. Anthropic Claude")
    print("4. OpenRouter")
    print("5. LM Studio (Local)")
    
    provider_map = {
        "1": "gemini",
        "2": "openai",
        "3": "anthropic",
        "4": "openrouter",
        "5": "lm_studio"
    }
    reverse_provider_map = {v: k for k, v in provider_map.items()}
    
    current_provider = current_config.get("provider", "gemini")
    current_choice = reverse_provider_map.get(current_provider, "1")
    
    while True:
        choice = input(f"Enter choice (1-4) [default: {current_choice} ({current_provider.capitalize()})]: ").strip()
        if not choice:
            choice = current_choice
        
        if choice in ["1", "2", "3", "4", "5"]:
            break
        print("Invalid choice. Please try again.")

    provider = provider_map[choice]
    
    # API Key
    while True:
        current_key = current_config.get("api_key", "") if current_config.get("provider") == provider else ""
        masked_key = "*" * 8 if current_key else "None"
        
        api_key = input(f"Enter your {provider.capitalize()} API Key [default: {masked_key}]: ").strip()
        
        if not api_key:
            if current_key:
                api_key = current_key
                print("Using existing API Key.")
                break
            else:
                print("API Key cannot be empty.")
                continue
        
        print("Validating API Key...")
        if validate_key(provider, api_key, current_config):
            print("API Key verified!")
            break
        else:
            print("Invalid API Key or connection error. Please try again.")
            retry = input("Retry? (y/n): ").lower().strip()
            if retry not in ['y', 'yes']:
                sys.exit(0)
    
    config = {
        "provider": provider,
        "api_key": api_key
    }
    
    # Model Selection
    if provider == "gemini":
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            print("Fetching available Gemini models...")
            models_pager = client.models.list()
            models = [m for m in models_pager]
            
            print("Available Models:")
            display_models = []
            for m in models:
                if "gemini" in m.name and "vision" not in m.name:
                     display_models.append(m)
            
            if not display_models:
                display_models = models

            current_model = current_config.get("model", "")
            default_index = 1
            
            for i, m in enumerate(display_models):
                marker = " *" if m.name == current_model else ""
                print(f"{i + 1}. {m.name} ({m.display_name}){marker}")
                if m.name == current_model:
                    default_index = i + 1
            
            while True:
                m_choice = input(f"Select a model (number) [default: {default_index}]: ").strip()
                if not m_choice:
                    m_choice = str(default_index)
                
                if m_choice.isdigit() and 1 <= int(m_choice) <= len(display_models):
                    model = display_models[int(m_choice) - 1].name
                    break
                print("Invalid choice.")
        except Exception as e:
            print(f"Error fetching models: {e}")
            model = current_config.get("model", "gemini-2.0-flash")
            print(f"Using default: {model}")
        
        config["model"] = model
    
    elif provider == "openrouter":
         current_model = current_config.get("model", "google/gemini-2.0-flash-001")
         model = input(f"Enter OpenRouter model [default: {current_model}]: ").strip()
         if not model:
             model = current_model
         config["model"] = model

    elif provider == "lm_studio":
        # Base URL
        default_url = "http://localhost:1234/v1"
        current_url = current_config.get("base_url", default_url)
        base_url = input(f"Enter LM Studio Base URL [default: {current_url}]: ").strip()
        if not base_url:
            base_url = current_url
        config["base_url"] = base_url
        
        # Dummy API Key
        config["api_key"] = "lm-studio"
        
        # Model Selection
        try:
            from openai import OpenAI
            client = OpenAI(base_url=base_url, api_key="lm-studio")
            print("Fetching available LM Studio models...")
            models = client.models.list()
            
            print("Available Models:")
            display_models = [m.id for m in models.data]
            
            if not display_models:
                display_models = ["local-model"]

            current_model = current_config.get("model", "")
            default_index = 1
            
            for i, m_name in enumerate(display_models):
                marker = " *" if m_name == current_model else ""
                print(f"{i + 1}. {m_name}{marker}")
                if m_name == current_model:
                    default_index = i + 1
            
            while True:
                m_choice = input(f"Select a model (number) [default: {default_index}]: ").strip()
                if not m_choice:
                    m_choice = str(default_index)
                
                if m_choice.isdigit() and 1 <= int(m_choice) <= len(display_models):
                    model = display_models[int(m_choice) - 1]
                    break
                print("Invalid choice.")
        except Exception as e:
            print(f"Error fetching models: {e}")
            model = current_config.get("model", "local-model")
            print(f"Using default: {model}")
        
        config["model"] = model

    save_config(config)
    print(f"Configuration saved to {CONFIG_FILE}")
    return config

def validate_key(provider, api_key, config=None):
    try:
        if provider == "gemini":
            from google import genai
            client = genai.Client(api_key=api_key)
            # Simple call to verify
            client.models.get(model='gemini-2.0-flash')
            return True
        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            client.models.list()
            return True
        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            try:
                client.messages.create(
                    model="claude-3-haiku-20240307", 
                    max_tokens=1, 
                    messages=[{"role": "user", "content": "hi"}]
                )
                return True
            except anthropic.AuthenticationError:
                return False
            except Exception:
                pass
            return True
        elif provider == "openrouter":
            from openai import OpenAI
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            client.models.list()
            return True
        elif provider == "lm_studio":
            from openai import OpenAI
            # Try to get URL from config, default to localhost
            base_url = "http://localhost:1234/v1"
            if config and "base_url" in config:
                base_url = config["base_url"]
            
            try:
                client = OpenAI(base_url=base_url, api_key="lm-studio")
                client.models.list()
                return True
            except Exception:
                # If we are in setup wizard and haven't set base_url yet, this might fail unless default works.
                # However, since we now ask for Base URL *after* API key validation (which we skip for LM Studio effectively by passing dummy),
                # we might just want to return True here and rely on the model fetching step in setup_wizard to validate connection.
                # But to be robust for *existing* configs, we should try.
                return True
    except Exception:
        return False
    return False

# --- Memory Management ---
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"static_facts": []}
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
            if "static_facts" not in data:
                data["static_facts"] = []
            return data
    except json.JSONDecodeError:
        return {"static_facts": []}

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=4)

def handle_slash_command(query, memory):
    cmd_parts = query.strip().split(" ", 1)
    command = cmd_parts[0].lower()
    args = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

    if command == "/remember":
        if not args:
            print("Usage: /remember <fact>")
        else:
            if args not in memory["static_facts"]:
                memory["static_facts"].append(args)
                save_memory(memory)
                print(f"Remembered: {args}")
            else:
                print("I already know that.")
        return True

    elif command == "/forget":
        if not args:
            print("Usage: /forget <fact>")
        else:
            if args in memory["static_facts"]:
                memory["static_facts"].remove(args)
                save_memory(memory)
                print(f"Forgot: {args}")
            else:
                print("I don't recall that fact.")
        return True

    elif command == "/mem":
        print("\n--- Persistent Memory ---")
        if not memory["static_facts"]:
            print("(Empty)")
        else:
            for fact in memory["static_facts"]:
                print(f"- {fact}")
        print("-------------------------")
        return True
    
    return False # Not a handled slash command

# --- AI Interaction ---
def get_system_prompt(memory, session_history):
    
    memory_section = ""
    if memory["static_facts"]:
        memory_section = "MEMORY (Always Context):\n" + "\n".join([f"- {f}" for f in memory["static_facts"]]) + "\n\n"

    history_section = ""
    if session_history:
        # Limit history to last 5 turns to save context
        display_history = session_history[-10:] 
        history_section = "SESSION HISTORY:\n"
        for turn in display_history:
             history_section += f"User: {turn['user']}\nAssistant: {turn['explanation']}\nCode: {turn['command']}\n"
        history_section += "\n"

    return f"""
You are a Windows PowerShell assistant. 
Your task is to translate the user's natural language request into a valid PowerShell command.

{memory_section}{history_section}
You MUST return a JSON object with exactly two keys:
1. "explanation": A brief explanation of what the command does.
2. "command": The exact PowerShell command to execute.

CAPABILITIES:
- You have access to real-time information via Google Search.
- **CRITICAL**: For any query regarding sports schedules, news, prices, or current events, you MUST perform a Google Search before generating the command.
- If the information is time-sensitive, do not rely on internal knowledge.

RESTRICTIONS:
- Do NOT output markdown code blocks. Output RAW JSON only.
- Ensure the JSON is valid.
- The command must be for PowerShell on Windows.
    """.strip()

def call_ai(provider, api_key, config, query, memory, session_history):
    system_prompt = get_system_prompt(memory, session_history)
    
    try:
        if provider == "gemini":
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=api_key)
            model_name = config.get("model", "gemini-2.0-flash")
            
            # Enable Google Search Grounding
            google_search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"{system_prompt}\n\nUser Request: {query}",
                    config=types.GenerateContentConfig(
                        tools=[google_search_tool],
                        response_modalities=["TEXT"]
                    )
                )
                
                # Check for grounding metadata
                if response.candidates and response.candidates[0].grounding_metadata and response.candidates[0].grounding_metadata.search_entry_point:
                     print(f"\033[94m[Search Grounding Used]\033[0m")
                
                return response.text
                             
            except Exception as e:
                # Fallback without search if it fails (e.g. model doesn't support it or quota exceeded)
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                     print(f"\033[93mSearch quota limit reached. Falling back to standard generation...\033[0m")
                else:
                     print(f"Search warning: {e}")
                response = client.models.generate_content(
                    model=model_name,
                    contents=f"{system_prompt}\n\nUser Request: {query}"
                )
                return response.text

        elif provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content

        elif provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": query}]
            )
            return message.content[0].text
        
        elif provider == "openrouter":
            from openai import OpenAI
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            # OpenRouter handling... (simplified for brevity, same as before)
            model = config.get("model", "google/gemini-2.0-flash-001")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ]
            )
            return response.choices[0].message.content

        elif provider == "lm_studio":
            from openai import OpenAI
            base_url = config.get("base_url", "http://localhost:1234/v1")
            client = OpenAI(base_url=base_url, api_key="lm-studio")
            
            model = config.get("model", "local-model")
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content

    except Exception as e:
        print(f"AI Error: {e}")
        return None

def parse_response(response_text):
    if not response_text: return None
    cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        print("Error: Failed to parse AI response.")
        return None

# --- Main Logic ---
def execute_command(command):
    print("Executing...")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command], 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        if result.stdout: print(result.stdout)
        if result.stderr: print(f"\033[91mError/Warning:\033[0m\n{result.stderr}")
        if result.returncode != 0: print(f"Command returned exit code: {result.returncode}")
    except Exception as e:
        print(f"Execution failed: {e}")

def process_query(query, config, memory, session_history):
    # Check for slash commands first
    if query.startswith("/"):
        if handle_slash_command(query, memory):
            return

    provider = config.get("provider")
    api_key = config.get("api_key")
    
    print(f"Thinking... ", end="", flush=True)
    response_text = call_ai(provider, api_key, config, query, memory, session_history)
    print("\r", end="") # Clear thinking line
    
    data = parse_response(response_text)
    if not data: return

    explanation = data.get("explanation", "No explanation.")
    command = data.get("command", "")
    
    print("-" * 40)
    print(f"\033[96mExplanation:\033[0m {explanation}")
    print(f"\033[93mCommand:\033[0m     {command}")
    print("-" * 40)

    # Add to history BEFORE execution? Or after? 
    # Let's add it now so subsequent turns know what was suggested.
    session_history.append({"user": query, "explanation": explanation, "command": command})

    while True:
        confirm = input("Execute? (y/n): ").lower().strip()
        if confirm in ['y', 'yes']:
            execute_command(command)
            break
        elif confirm in ['n', 'no']:
            print("Aborted.")
            break

def main():
    parser = argparse.ArgumentParser(description="Smart Agent")
    parser.add_argument("query", nargs="*", help="Query")
    parser.add_argument("--config", action="store_true", help="Reset config")
    args = parser.parse_args()

    config = load_config()
    if args.config or config is None:
        config = setup_wizard()
        if not args.query:
             # Stay in main loop if configured without query? 
             pass

    memory = load_memory()
    session_history = []

    # Single Shot Mode
    if args.query:
        query = " ".join(args.query)
        process_query(query, config, memory, session_history)
    
    # Interactive Loop (REPL)
    else:
        print("\033[92mSmart Agent Interactive Mode\033[0m")
        print("Type 'exit' to quit. Use '/remember <fact>' to add memory.")
        while True:
            try:
                user_input = input("\n> ").strip()
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            
            if not user_input: continue
            if user_input.lower() in ["exit", "quit"]:
                break
            
            process_query(user_input, config, memory, session_history)

if __name__ == "__main__":
    main()
