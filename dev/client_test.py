#!/usr/bin/env python3

import argparse
import numpy as np
import tritonclient.grpc as grpcclient
import sys
import os

# --- Helper function to create inputs correctly ---
def create_audio_input(audio_bytes):
    """
    Creates the gRPC input tensor for audio bytes with the correct
    2D shape (batch_size, 1) that Triton expects.
    """
    # FIX #1: The audio bytes must be wrapped in a 2D numpy array.
    # The shape becomes (1, 1) for a single audio file in the batch.
    audio_bytes_np = np.array([[audio_bytes]], dtype=np.object_)

    # The shape [1, 1] tells Triton: batch_size=1, input_dims=1
    input_audio = grpcclient.InferInput("AUDIO_BYTES", [1, 1], "BYTES")
    input_audio.set_data_from_numpy(audio_bytes_np)
    return input_audio


def test_asr(client, args):
    """Tests the ASR model in isolation."""
    model_name = "asr"
    print(f"--- Testing Model: {model_name} ---")
    with open(args.audio_file, "rb") as f:
        audio_bytes = f.read()

    input_audio = create_audio_input(audio_bytes)
    outputs = [grpcclient.InferRequestedOutput("TRANSCRIPTIONS")]

    results = client.infer(model_name=model_name, inputs=[input_audio], outputs=outputs)
    transcriptions = results.as_numpy("TRANSCRIPTIONS")
    print("\n[ASR Model Output]")
    for line in transcriptions:
        print(line.decode("utf-8"))


def test_diarizer(client, args):
    """Tests the Diarizer model in isolation."""
    model_name = "diarizer"
    print(f"--- Testing Model: {model_name} ---")
    with open(args.audio_file, "rb") as f:
        audio_bytes = f.read()

    input_audio = create_audio_input(audio_bytes)
    outputs = [grpcclient.InferRequestedOutput("DIARIZATION_RTTM")]

    results = client.infer(model_name=model_name, inputs=[input_audio], outputs=outputs)
    rttm_lines = results.as_numpy("DIARIZATION_RTTM")
    print("\n[Diarizer Model Output (RTTM)]")
    for line in rttm_lines:
        print(line.decode("utf-8"))


def test_stitcher(client, args):
    """Tests the Stitcher model with mock data."""
    model_name = "stitcher"
    print(f"--- Testing Model: {model_name} (with mock data) ---")
    
    # Mock ASR data
    mock_asr_data = np.array([
        b"[0.50] - [2.50]: hello world",
        b"[3.00] - [5.00]: this is a test"
    ], dtype=np.object_)
    
    # Mock RTTM data
    mock_rttm_data = np.array([
        b"SPEAKER <NA> 1 0.40 2.20 <NA> <NA> speaker_0 <NA> <NA>",
        b"SPEAKER <NA> 1 2.90 2.30 <NA> <NA> speaker_1 <NA> <NA>"
    ], dtype=np.object_)

    # Create inputs
    input_asr = grpcclient.InferInput("ASR_TRANSCRIPT", mock_asr_data.shape, "BYTES")
    input_asr.set_data_from_numpy(mock_asr_data)
    
    input_rttm = grpcclient.InferInput("DIARIZATION_RTTM", mock_rttm_data.shape, "BYTES")
    input_rttm.set_data_from_numpy(mock_rttm_data)

    outputs = [grpcclient.InferRequestedOutput("FINAL_TRANSCRIPT")]

    results = client.infer(model_name=model_name, inputs=[input_asr, input_rttm], outputs=outputs)
    final_transcript = results.as_numpy("FINAL_TRANSCRIPT")
    print("\n[Stitcher Model Output]")
    for line in final_transcript:
        print(line.decode("utf-8"))


def test_ensemble(client, args):
    """Tests the full end-to-end ensemble model."""
    model_name = "asr_diarization_ensemble"
    print(f"--- Testing Model: {model_name} ---")
    with open(args.audio_file, "rb") as f:
        audio_bytes = f.read()

    input_audio = create_audio_input(audio_bytes)
    outputs = [grpcclient.InferRequestedOutput("FINAL_TRANSCRIPT")]

    results = client.infer(model_name=model_name, inputs=[input_audio], outputs=outputs)
    final_transcript = results.as_numpy("FINAL_TRANSCRIPT")
    print("\n[Final Ensemble Output]")
    for line in final_transcript:
        print(line.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="gRPC client for testing the ASR+Diarization pipeline.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-u", "--url", type=str, required=False, default="localhost:10001",
        help="Inference server URL. Default is localhost:8001."
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command help')

    # Sub-parser for ASR
    parser_asr = subparsers.add_parser('asr', help='Test the ASR model individually.')
    parser_asr.add_argument('audio_file', type=str, help='Path to the audio file.')
    parser_asr.set_defaults(func=test_asr)

    # Sub-parser for Diarizer
    parser_diarizer = subparsers.add_parser('diarizer', help='Test the diarizer model individually.')
    parser_diarizer.add_argument('audio_file', type=str, help='Path to the audio file.')
    parser_diarizer.set_defaults(func=test_diarizer)

    # Sub-parser for Stitcher
    parser_stitcher = subparsers.add_parser('stitcher', help='Test the stitcher model with mock data.')
    parser_stitcher.set_defaults(func=test_stitcher)

    # Sub-parser for Ensemble
    parser_ensemble = subparsers.add_parser('ensemble', help='Test the full end-to-end pipeline.')
    parser_ensemble.add_argument('audio_file', type=str, help='Path to the audio file.')
    parser_ensemble.set_defaults(func=test_ensemble)

    args = parser.parse_args()

    try:
        triton_client = grpcclient.InferenceServerClient(url=args.url, verbose=False)
    except Exception as e:
        print(f"Client creation failed: {e}")
        sys.exit(1)

    # --- Server Health Checks ---
    try:
        if not triton_client.is_server_live():
            print(f"Triton server at {args.url} is not live.")
            sys.exit(1)
        if not triton_client.is_server_ready():
            print(f"Triton server at {args.url} is not ready.")
            sys.exit(1)
        print(f"Successfully connected to Triton server at {args.url}")
    except grpcclient.InferenceServerException as e:
        print(f"Could not check server health: {e}")
        sys.exit(1)

    # --- Run the selected test function ---
    try:
        args.func(triton_client, args)
    except grpcclient.InferenceServerException as e:
        # FIX #2: Simply print the exception object. It contains the full error string.
        print(f"\nINFERENCE FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected client-side error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

