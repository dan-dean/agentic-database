from typing import List, Dict, Any
import sys
import json
import shutil


class LLMAssistant:

    def clear_lines(num_lines):
        """Move the cursor up num_lines and clear those lines."""
        for _ in range(num_lines):
            sys.stdout.write("\033[1A")  # Move cursor up one line
            sys.stdout.write("\033[2K")  # Clear entire line

    def get_num_lines(text):
        """Calculate the number of terminal lines the text occupies."""
        terminal_width = shutil.get_terminal_size().columns
        lines = text.split("\n")
        num_lines = 0
        for line in lines:
            # Calculate the number of terminal lines for each line
            line_length = len(line)
            num_terminal_lines = max(
                1, (line_length + terminal_width - 1) // terminal_width
            )
            num_lines += num_terminal_lines
        return num_lines

    def get_structured_output(
        model,
        messages: List[Dict[str, str]],
        response_schema: Dict[str, Any],
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Streams the model output, printing the output progressively,
        and returns the accumulated data as a dictionary after parsing it as JSON.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            response_schema (Dict[str, Any]): The JSON schema defining the expected output.
            verbose (bool): If True, prints the streaming output.

        Returns:
            Dict[str, Any]: The accumulated data as a dictionary after parsing the complete JSON.
        """
        response = model.create_chat_completion(
            messages=messages,
            response_format={"type": "json", "schema": response_schema},
            stream=True,
        )

        accumulated_text = ""
        previous_num_lines = 0

        for chunk in response:
            # Extract the text from the streamed chunk
            if (
                "choices" not in chunk
                or not chunk["choices"]
                or "delta" not in chunk["choices"][0]
                or "content" not in chunk["choices"][0]["delta"]
            ):
                continue
            chunk_text = chunk["choices"][0]["delta"]["content"]

            # Concatenate the extracted text
            accumulated_text += chunk_text

            if verbose:
                # Print the new chunk without clearing previous output
                sys.stdout.write(chunk_text)
                sys.stdout.flush()

        if verbose:
            sys.stdout.write("\n")

        # After streaming is complete, parse the accumulated text as JSON
        try:
            accumulated_data = json.loads(accumulated_text)
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            accumulated_data = {}

        return accumulated_data
