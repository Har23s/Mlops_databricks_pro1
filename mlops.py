import time

# 1. Get selections
selected_provider = provider_dropdown.value
selected_model = model_dropdown.value

print("\n Chat session started.")
print(f"   Using Provider: {selected_provider}")
print(f"   Using Model: {selected_model}")
print("   Type 'exit' to end the session.\n")

turn_counter = 1
conversation_history = []  # Store tuples: (role, text)

# 2. Start one MLflow run for the entire chat session
session_run_name = f"Chat Session - {int(time.time())}"
with mlflow.start_run(run_name=session_run_name) as run:

    mlflow.log_param("provider", selected_provider)
    mlflow.log_param("model_used", selected_model)

    print(f" MLflow Run Started. View traces at: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")

    while True:
        my_prompt = input("You: ")

        if my_prompt.lower() == "exit":
            print("\n Assistant: Goodbye! Chat session ended.")
            # Save final conversation as an artifact
            chat_log_text = "\n".join(
                f"{role.capitalize()}: {text}" for role, text in conversation_history
            )
            mlflow.log_text(chat_log_text, artifact_file="full_conversation.txt")
            break

        conversation_history.append(("you", my_prompt))

        # Build the full context string for the model
        full_context = "\n".join(
            f"{role.capitalize()}: {text}" for role, text in conversation_history
        )

        # 3. Call the unified AI function
        response_text = get_ai_response(
            provider=selected_provider,
            model=selected_model,
            prompt=full_context
        )

        conversation_history.append(("assistant", response_text))

        print(f"🤖 Assistant: {response_text}\n")

        # Log response length metric
        mlflow.log_metric(f"response_length_turn_{turn_counter}", len(response_text))

        # 🔹 Save both prompt + response as artifact for this turn
        turn_log_text = f"You: {my_prompt}\nAssistant: {response_text}"
        mlflow.log_text(turn_log_text, artifact_file=f"turn_{turn_counter}.txt")

        turn_counter += 1
