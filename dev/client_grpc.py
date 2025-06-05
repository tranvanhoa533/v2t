#!/usr/bin/env python3

import argparse
import numpy as np
import tritonclient.grpc as grpcclient
# from tritonclient.utils import np_to_triton_dtype # Not strictly needed for this client's usage
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="gRPC client for Triton ChunkFormer model.")
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:10001", # Default gRPC port
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=False,
        default="v2t_vi",
        help="Name of the model to use in Triton. Default is chunkformer.",
    )
    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to the audio file to transcribe (e.g., .wav, .mp3).",
    )
    parser.add_argument(
        "--audio_format",
        type=str,
        default=None, # Server will use its configured default if this is not sent
        help="Format of the audio file (e.g., 'wav', 'mp3'). Important for the server to know how to decode the bytes."
    )
    parser.add_argument(
        "--total_batch_duration_sec",
        type=int,
        default=None,
        help="Total audio duration (in seconds) in a batch. (Optional, server default: 1800)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Size of the chunks for processing. (Optional, server default: 64)"
    )
    parser.add_argument(
        "--left_context_size",
        type=int,
        default=None,
        help="Size of the left context for attention. (Optional, server default: 128)"
    )
    parser.add_argument(
        "--right_context_size",
        type=int,
        default=None,
        help="Size of the right context for attention. (Optional, server default: 128)"
    )
    parser.add_argument(
        "--autocast_dtype_str",
        type=str,
        default=None,
        choices=["fp32", "bf16", "fp16", "NONE"],
        help="Autocast dtype for server-side processing ('fp32', 'bf16', 'fp16', 'NONE'). (Optional, server default: 'NONE')"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose client output.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        sys.exit(1)

    try:
        # Create gRPC client
        triton_client = grpcclient.InferenceServerClient(
            url=args.url, verbose=args.verbose
        )
    except Exception as e:
        print("Client creation failed: " + str(e))
        sys.exit(1)

    # Check server and model health (optional but good practice)
    if not triton_client.is_server_live():
        print(f"Triton server at {args.url} is not live.")
        sys.exit(1)
    if not triton_client.is_server_ready(): # Checks if server is ready for inference
        print(f"Triton server at {args.url} is not ready.")
        sys.exit(1)
    try:
        if not triton_client.is_model_ready(args.model_name): # Checks if the specific model is ready
            print(f"Model {args.model_name} is not ready on Triton server at {args.url}.")
            sys.exit(1)
    except grpcclient.InferenceServerException as e:
         print(f"Could not check model readiness for {args.model_name}: {e}")
         sys.exit(1)


    inputs = []
    outputs = []

    # Read audio file content as bytes
    with open(args.audio_file, "rb") as f:
        audio_bytes = f.read()

    # Input for AUDIO_BYTES
    # For gRPC, when the server expects bytes via TYPE_STRING,
    # we send it as a NumPy array of objects, where each object is bytes.
    # The shape [1] indicates a single audio file in this request.
    audio_bytes_np = np.array([audio_bytes], dtype=np.object_)
    input_audio = grpcclient.InferInput("AUDIO_BYTES", [1], "BYTES") # Client uses "BYTES" for this
    input_audio.set_data_from_numpy(audio_bytes_np)
    inputs.append(input_audio)

    # Optional input: AUDIO_FORMAT
    if args.audio_format is not None:
        audio_format_np = np.array([args.audio_format.lower().encode('utf-8')], dtype=np.object_)
        input_format = grpcclient.InferInput("AUDIO_FORMAT", [1], "BYTES") # String inputs are also "BYTES" type
        input_format.set_data_from_numpy(audio_format_np)
        inputs.append(input_format)
        if args.verbose:
            print(f"Client: Sending audio_format: {args.audio_format.lower()}")

    # Other Optional inputs
    if args.total_batch_duration_sec is not None:
        input_tensor = grpcclient.InferInput("TOTAL_BATCH_DURATION_SEC", [1], "INT32")
        input_tensor.set_data_from_numpy(np.array([args.total_batch_duration_sec], dtype=np.int32))
        inputs.append(input_tensor)

    if args.chunk_size is not None:
        input_tensor = grpcclient.InferInput("CHUNK_SIZE", [1], "INT32")
        input_tensor.set_data_from_numpy(np.array([args.chunk_size], dtype=np.int32))
        inputs.append(input_tensor)

    if args.left_context_size is not None:
        input_tensor = grpcclient.InferInput("LEFT_CONTEXT_SIZE", [1], "INT32")
        input_tensor.set_data_from_numpy(np.array([args.left_context_size], dtype=np.int32))
        inputs.append(input_tensor)

    if args.right_context_size is not None:
        input_tensor = grpcclient.InferInput("RIGHT_CONTEXT_SIZE", [1], "INT32")
        input_tensor.set_data_from_numpy(np.array([args.right_context_size], dtype=np.int32))
        inputs.append(input_tensor)

    if args.autocast_dtype_str is not None:
        autocast_str_np = np.array([args.autocast_dtype_str.encode('utf-8')], dtype=np.object_)
        input_tensor = grpcclient.InferInput("AUTOCAST_DTYPE_STR", [1], "BYTES")
        input_tensor.set_data_from_numpy(autocast_str_np)
        inputs.append(input_tensor)

    # Define the requested output
    outputs.append(grpcclient.InferRequestedOutput("TRANSCRIPTIONS"))

    if args.verbose:
        print(f"Client: Sending gRPC request to Triton ({args.url}) for model '{args.model_name}'...")
        print(f"Client: Audio file: '{args.audio_file}' ({len(audio_bytes)} bytes)")

    try:
        # Perform inference
        results = triton_client.infer(
            model_name=args.model_name,
            inputs=inputs,
            outputs=outputs,
            # request_id=str(uuid.uuid4()) # Optional: for tracing
        )
    except grpcclient.InferenceServerException as e:
        print(f"Client: gRPC Inference failed: {e.message()} (details: {e.details()})")
        sys.exit(1)
    except ConnectionRefusedError:
        print(f"Client: gRPC Connection refused. Ensure Triton Inference Server is running at {args.url} and the gRPC port is open.")
        sys.exit(1)
    except Exception as e:
        print(f"Client: An unexpected error occurred during inference: {e}")
        sys.exit(1)

    # Get the output
    # For TYPE_STRING, as_numpy() returns an array of bytes objects
    transcriptions_bytes_np = results.as_numpy("TRANSCRIPTIONS")

    if transcriptions_bytes_np is not None:
        print("\nTranscription Results:")
        for line_bytes in transcriptions_bytes_np:
            print(line_bytes.decode("utf-8")) # Decode bytes to string
    else:
        print("No transcriptions received from the server.")

if __name__ == "__main__":
    main()
