#!/bin/bash

SESSION_NAME="wmt_eval"

# Array of models (the registry key of each provider config)
SYSTEMS=(
    "command-a-plus-05-2026"
    "command-r7b-12-2024"
    "c4ai-aya-expanse-32b"
    "deepseek-ai/DeepSeek-V3"
    "Qwen/Qwen3-235B-A22B-fp8-tput"
    "Qwen/Qwen2.5-7B-Instruct-Turbo"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    "meta-llama/Llama-4-Scout-17B-16E-Instruct"
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "gpt-5.1"
    "claude-sonnet-4-5-20250929"
    "mistral-medium-3.5"
    "gemini-3.1-pro-preview"
    "gemma-3-12b-it"
    "gemma-3-27b-it"
    "DeepL"
    "GoogleTranslate"
    "YandexTranslate"
    "MicrosoftTranslator"
)

# Function to sanitize window name
sanitize_name() {
    echo "$1" | tr -c '[:alnum:]' '_'
}

# Create a new tmux session with the first system
first_system="${SYSTEMS[0]}"
window_name=$(sanitize_name "$first_system")
tmux new-session -d -s $SESSION_NAME
tmux rename-window -t $SESSION_NAME "$window_name"
tmux send-keys -t $SESSION_NAME "conda activate wmt; python main.py --model '${first_system}'" C-m

# Create a new window for each remaining system
for system in "${SYSTEMS[@]:1}"; do
    # Create a new window with sanitized name
    window_name=$(sanitize_name "$system")
    tmux new-window -t $SESSION_NAME -n "$window_name"
    tmux send-keys -t $SESSION_NAME:"$window_name" "conda activate wmt; python main.py --model '${system}'" C-m
done

# Select the first window
tmux select-window -t $SESSION_NAME:0

# Attach to the session
tmux attach -t $SESSION_NAME
