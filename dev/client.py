#!/usr/bin/env python3

import argparse
import numpy as np
import tritonclient.http as httpclient
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:10000",
        help="Inference server URL. Default is localhost:8000.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=False,
        default="v2t_vi",
        help="Name of the model to use. Default is chunkformer.",
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--total_batch_duration_sec",
        type=int,
        default=14400, # Server will use its default if None
        help="Total audio duration (in seconds) in a batch. (Optional, server default: 1800)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None, # Server will use its default if None
        help="Size of the chunks. (Optional, server default: 64)"
    )
    parser.add_argument(
        "--left_context_size",
        type=int,
        default=None, # Server will use its default if None
        help="Size of the left context. (Optional, server default: 128)"
    )
    parser.add_argument(
        "--right_context_size",
        type=int,
        default=None, # Server will use its default if None
        help="Size of the right context. (Optional, server default: 128)"
    )
    parser.add_argument(
        "--autocast_dtype_str",
        type=str,
        default=None, # Server will use its default if None
        choices=["fp32", "bf16", "fp16", "NONE"],
        help="Autocast dtype ('fp32', 'bf16', 'fp16', 'NONE'). (Optional, server default: 'NONE')"
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        sys.exit(1)

    try:
        triton_client = httpclient.InferenceServerClient(url=args.url, verbose=False)
    except Exception as e:
        print("Client creation failed: " + str(e))
        sys.exit(1)

    inputs = []
    outputs = []

    # Read audio file content as bytes
    with open(args.audio_file, "rb") as f:
        audio_bytes = f.read()

    # Input for AUDIO_BYTES
    # For TYPE_STRING in config.pbtxt that expects bytes, pass as numpy object array
    audio_bytes_np = np.array([audio_bytes], dtype=np.object_)
    inputs.append(httpclient.InferInput("AUDIO_BYTES", [1], "BYTES")) # Datatype "BYTES" for client, maps to TYPE_STRING on server
    inputs[-1].set_data_from_numpy(audio_bytes_np)

    # Optional inputs: Only add them if a value is provided by the user
    if args.total_batch_duration_sec is not None:
        input_tensor = httpclient.InferInput("TOTAL_BATCH_DURATION_SEC", [1], "INT32")
        input_tensor.set_data_from_numpy(np.array([args.total_batch_duration_sec], dtype=np.int32))
        inputs.append(input_tensor)

    if args.chunk_size is not None:
        input_tensor = httpclient.InferInput("CHUNK_SIZE", [1], "INT32")
        input_tensor.set_data_from_numpy(np.array([args.chunk_size], dtype=np.int32))
        inputs.append(input_tensor)

    if args.left_context_size is not None:
        input_tensor = httpclient.InferInput("LEFT_CONTEXT_SIZE", [1], "INT32")
        input_tensor.set_data_from_numpy(np.array([args.left_context_size], dtype=np.int32))
        inputs.append(input_tensor)

    if args.right_context_size is not None:
        input_tensor = httpclient.InferInput("RIGHT_CONTEXT_SIZE", [1], "INT32")
        input_tensor.set_data_from_numpy(np.array([args.right_context_size], dtype=np.int32))
        inputs.append(input_tensor)

    if args.autocast_dtype_str is not None:
        input_tensor = httpclient.InferInput("AUTOCAST_DTYPE_STR", [1], "BYTES") # Datatype "BYTES" for string inputs
        input_tensor.set_data_from_numpy(np.array([args.autocast_dtype_str.encode('utf-8')], dtype=np.object_))
        inputs.append(input_tensor)


    # Define the output
    outputs.append(httpclient.InferRequestedOutput("TRANSCRIPTIONS"))

    print(f"Sending request to Triton for model '{args.model_name}' with audio file '{args.audio_file}'...")

    try:
        results = triton_client.infer(
            model_name=args.model_name, inputs=inputs, outputs=outputs
        )
    except httpclient.InferenceServerException as e:
        print("Inference failed: " + str(e))
        if "Unable to parse request body" in str(e) or "unexpected end of string" in str(e):
             print("This might be due to a very large audio file. Consider limitations on HTTP request sizes if applicable.")
        sys.exit(1)
    except ConnectionRefusedError:
        print(f"Connection refused. Ensure Triton Inference Server is running at {args.url}.")
        sys.exit(1)


    # Get the output
    transcriptions_np = results.as_numpy("TRANSCRIPTIONS")

    if transcriptions_np is not None:
        print("\nTranscription Results:")
        for line_bytes in transcriptions_np:
            # The output from server (np.array(..., dtype=object)) might contain bytes that need decoding
            if isinstance(line_bytes, bytes):
                print(line_bytes.decode("utf-8"))
            else: # Should already be string if server model.py did it right
                print(line_bytes)
    else:
        print("No transcriptions received.")

if __name__ == "__main__":
    main()