import json
import os
import torch
import torchaudio
import io
import yaml
import jiwer # Though not used in endless_decode, part of original imports
import pandas as pd # Though not used in endless_decode, part of original imports
from contextlib import nullcontext
import tempfile
import numpy as np
import triton_python_backend_utils as pb_utils

# Imports from the 'model' directory (copied from chunkformer-main)
from model_code.utils.init_model import init_model
from model_code.utils.checkpoint import load_checkpoint # Part of init_model implicitly
from model_code.utils.file_utils import read_symbol_table # Part of init_model implicitly
from model_code.utils.ctc_utils import get_output_with_timestamps, get_output
import torchaudio.compliance.kaldi as kaldi
from pydub import AudioSegment


class TritonPythonModel:
    """
    Your Python model class for Triton.
    """

    def initialize(self, args):
        """
        Initialize the model. This function is called once when the model is loaded.
        """
        self.model_config = json.loads(args['model_config'])
        self.model_repository_path = args['model_repository']
        self.model_version = args['model_version']
        
        self.checkpoint_dir = os.path.join(self.model_repository_path, str(self.model_version), 'checkpoints')

        # Get device from parameters
        device_param = self.model_config['parameters'].get('DEVICE', {}).get('string_value', 'cuda')
        self.device = torch.device(device_param if torch.cuda.is_available() and device_param == "cuda" else "cpu")
        
        # Load model and char_dict
        self.model, self.char_dict = self._load_model_internal(self.checkpoint_dir, self.device)

        # Store default values from config.pbtxt parameters
        self.default_audio_format = self.model_config['parameters'].get('DEFAULT_AUDIO_FORMAT', {}).get('string_value', 'wav')
        self.default_total_batch_duration = int(self.model_config['parameters'].get('DEFAULT_TOTAL_BATCH_DURATION_SEC', {}).get('string_value', '1800'))
        self.default_chunk_size = int(self.model_config['parameters'].get('DEFAULT_CHUNK_SIZE', {}).get('string_value', '64'))
        self.default_left_context_size = int(self.model_config['parameters'].get('DEFAULT_LEFT_CONTEXT_SIZE', {}).get('string_value', '128'))
        self.default_right_context_size = int(self.model_config['parameters'].get('DEFAULT_RIGHT_CONTEXT_SIZE', {}).get('string_value', '128'))
        self.default_autocast_dtype_str = self.model_config['parameters'].get('DEFAULT_AUTOCAST_DTYPE_STR', {}).get('string_value', 'NONE')
        self.default_max_silence_frames = int(self.model_config['parameters'].get('DEFAULT_MAX_SILENCE_FRAMES', {}).get('string_value', '6'))
        
        self.autocast_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "NONE": None            
        }

        self.logger = pb_utils.Logger
        self.logger.log_info(f"Model initialized on device: {self.device}")
        self.logger.log_info(f"Checkpoint directory: {self.checkpoint_dir}")
        self.logger.log_info(f"Default audio format: {self.default_audio_format}")
        
        


    def _load_model_internal(self, model_checkpoint_path, device):
        """
        Loads the model using the logic from decode.py's init function.
        """
        config_path = os.path.join(model_checkpoint_path, "config.yaml")
        checkpoint_file_path = os.path.join(model_checkpoint_path, "pytorch_model.bin")
        symbol_table_path = os.path.join(model_checkpoint_path, "vocab.txt")

        if not os.path.exists(config_path):
            raise pb_utils.TritonModelException(f"Config file not found: {config_path}")
        if not os.path.exists(checkpoint_file_path):
            raise pb_utils.TritonModelException(f"Model checkpoint file not found: {checkpoint_file_path}")
        if not os.path.exists(symbol_table_path):
            raise pb_utils.TritonModelException(f"Symbol table not found: {symbol_table_path}")

        with open(config_path, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        
        model = init_model(config, model_checkpoint_path) # Pass base path for potential relative paths in config
        model.eval()
        load_checkpoint(model, checkpoint_file_path)

        model = model.to(device) # Move model to device

        symbol_table = read_symbol_table(symbol_table_path)
        char_dict = {v: k for k, v in symbol_table.items()}
        
        return model, char_dict

    # Inside TritonPythonModel class in model.py

    # Inside TritonPythonModel class in model.py
    # Ensure 'from pydub import AudioSegment' is present

    # Using the pydub version that worked for you
    def _load_audio_internal(self, audio_bytes_content, audio_format="wav"): # audio_format is key
        self.logger.log_info(f"Attempting to load audio from in-memory bytes using pydub. Specified format: {audio_format}")
        try:
            audio_file_like_object = io.BytesIO(audio_bytes_content)
            
            audio = AudioSegment.from_file(audio_file_like_object, format=audio_format) # Use the passed format
            self.logger.log_info(f"pydub: Loaded from bytes. Duration: {audio.duration_seconds:.2f}s, Channels: {audio.channels}, Frame Rate: {audio.frame_rate}")

            audio = audio.set_frame_rate(16000)
            audio = audio.set_sample_width(2)
            audio = audio.set_channels(1)
            self.logger.log_info(f"pydub: Resampled/Reformatted. Frame Rate: {audio.frame_rate}, Channels: {audio.channels}")
            
            samples_array = audio.get_array_of_samples()
            samples = torch.as_tensor(samples_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            self.logger.log_info(f"pydub: Output samples tensor shape: {samples.shape}, dtype: {samples.dtype}, device: {samples.device}")
            if samples.numel() > 0:
                 self.logger.log_info(f"pydub: Samples stats: min={samples.min()}, max={samples.max()}")
            return samples
        except Exception as e:
            self.logger.log_error(f"Detailed error in _load_audio_internal (pydub from bytes, format: {audio_format}): {e}")
            # import traceback; self.logger.log_error(traceback.format_exc())
            raise pb_utils.TritonModelException(f"Error loading audio from in-memory bytes with pydub (format: {audio_format}): {e}")
        
        
    @torch.no_grad()
    def _endless_decode_internal(self, current_args, model, char_dict):
        """
        Core decoding logic from decode.py's endless_decode, modified to return results.
        """
        device = model.encoder.embed.conv[0].weight.device # Get device from model
        # audio_path = current_args.long_form_audio
        
        subsampling_factor = model.encoder.embed.subsampling_factor
        chunk_size = current_args.chunk_size
        left_context_size = current_args.left_context_size
        right_context_size = current_args.right_context_size
        conv_lorder = model.encoder.cnn_module_kernel // 2

        max_length_limited_context = current_args.total_batch_duration
        max_length_limited_context = int((max_length_limited_context // 0.01)) // 2

        multiply_n = max_length_limited_context // chunk_size // subsampling_factor
        truncated_context_size = chunk_size * multiply_n

        def get_max_input_context(c, r, n): # Helper from original script
            return r + max(c, r) * (n - 1)

        rel_right_context_size = get_max_input_context(chunk_size, max(right_context_size, conv_lorder), model.encoder.num_blocks)
        rel_right_context_size = rel_right_context_size * subsampling_factor

        # waveform = self._load_audio_internal(audio_path)
        audio_bytes_data = current_args.audio_bytes_data 
        current_audio_format = current_args.audio_format # Get format from args

        self.logger.log_info(f"Decoding with format: {current_audio_format}")
        waveform = self._load_audio_internal(audio_bytes_data, audio_format=current_audio_format)
        
        waveform = waveform.to(device) # Ensure waveform is on the correct device
        self.logger.log_info(f"Waveform tensor for fbank: shape={waveform.shape}, device={waveform.device}, dtype={waveform.dtype}")
        
        offset_tensor = torch.zeros(1, dtype=torch.int, device=device)

        xs = kaldi.fbank(waveform,
                         num_mel_bins=80,
                         frame_length=25,
                         frame_shift=10,
                         dither=0.0,
                         energy_floor=0.0,
                         sample_frequency=16000).unsqueeze(0)
        xs = xs.to(device) # Ensure fbank features are on the correct device

        self.logger.log_info(f"FBank features (xs): shape={xs.shape}, device={xs.device}, dtype={xs.dtype}")
        if xs.numel() > 0:
            self.logger.log_info(f"FBank features stats: min={xs.min()}, max={xs.max()}, sum={xs.sum()}")
            if torch.isnan(xs).any() or torch.isinf(xs).any():
                self.logger.log_error("FBank features contain NaN or Inf values!")
        else:
            self.logger.log_error("FBank features are empty!")
            return [] # Early exit if features are empty
        
        hyps_list = []
        # Initialize caches on the correct device
        att_cache = torch.zeros((model.encoder.num_blocks, left_context_size, model.encoder.attention_heads, model.encoder._output_size * 2 // model.encoder.attention_heads), device=device)
        cnn_cache = torch.zeros((model.encoder.num_blocks, model.encoder._output_size, conv_lorder), device=device)
        
        self.logger.log_info(f"Starting decoding loop for audio. Total fbank frames: {xs.shape[1]}")
        self.logger.log_info(f"Loop parameters: truncated_context_size={truncated_context_size}, subsampling_factor={subsampling_factor}, step_size={truncated_context_size * subsampling_factor}")
        for idx, _ in enumerate(range(0, xs.shape[1], truncated_context_size * subsampling_factor)):
            self.logger.log_info(f"--- Decoding Loop Iteration: {idx} ---")
            start = max(truncated_context_size * subsampling_factor * idx, 0)
            end = min(truncated_context_size * subsampling_factor * (idx + 1) + 7, xs.shape[1])

            x_chunk = xs[:, start:end + rel_right_context_size]
            x_len = torch.tensor([x_chunk.shape[1]], dtype=torch.int, device=device) # x_chunk[0].shape[0] -> x_chunk.shape[1]
            
            self.logger.log_info(f"Loop {idx}: Fbank slice for chunk: start_frame={start}, end_frame_with_context={end + rel_right_context_size}")
            self.logger.log_info(f"Loop {idx}: x_chunk shape: {x_chunk.shape}, x_len: {x_len.item()}")
            
            if x_chunk.shape[1] == 0:
                self.logger.log_warn(f"Loop {idx}: x_chunk is empty, skipping model forward.")
                if start >= xs.shape[1] : break # Break if we are beyond the actual data
                continue

            encoder_outs, encoder_lens, _, att_cache, cnn_cache, offset_tensor = model.encoder.forward_parallel_chunk(
                xs=x_chunk,
                xs_origin_lens=x_len,
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                right_context_size=right_context_size,
                att_cache=att_cache,
                cnn_cache=cnn_cache,
                truncated_context_size=truncated_context_size,
                offset=offset_tensor
            )
            encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[:, :encoder_lens[0]] # Use encoder_lens[0]
            
            if chunk_size * multiply_n * subsampling_factor * idx + rel_right_context_size < xs.shape[1]:
                 # Make sure not to slice beyond actual length for the last chunk based on encoder_lens
                effective_truncated_size = min(truncated_context_size, encoder_outs.shape[1])
                encoder_outs = encoder_outs[:, :effective_truncated_size]

            offset_tensor = offset_tensor - encoder_lens + encoder_outs.shape[1]

            self.logger.log_info(f"Loop {idx}: encoder_outs shape (raw): {encoder_outs.shape}, encoder_lens: {encoder_lens.tolist()}")
            
            # Ensure encoder_lens is not empty and contains valid lengths
            if encoder_lens.numel() == 0 or encoder_lens[0] <= 0:
                self.logger.log_warn(f"Loop {idx}: Invalid encoder_lens ({encoder_lens.tolist()}), skipping CTC forward for this chunk.")
                # Check break condition based on original xs length
                if start + truncated_context_size * subsampling_factor >= xs.shape[1]:
                    self.logger.log_info(f"Loop {idx}: Reached end of input data based on original xs length. Breaking loop.")
                    break
                continue
            
            hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
            hyps_list.append(hyp)
            
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            if start + truncated_context_size * subsampling_factor >= xs.shape[1]: # Adjusted break condition
                 break
        
        if not hyps_list: # Handle cases where audio is too short / no hyps generated
            return []

        hyps_combined = torch.cat(hyps_list)
        decode_results = get_output_with_timestamps([hyps_combined], char_dict, current_args.max_silence_frames)[0]
        self.logger.log_info(f"Results from get_output_with_timestamps (before formatting): {decode_results}")

        output_strings = []
        if decode_results:
            for item in decode_results:
                output_strings.append(f"[{item['start']}] - [{item['end']}]: {item['decode']}")
        else:
            self.logger.log_warn("decode_results from get_output_with_timestamps is empty.")
        
        self.logger.log_info(f"Final formatted output_strings: {output_strings}")
        
        return output_strings


    # Inside model.py, class TritonPythonModel:

    def execute(self, requests):
        # THIS IS THE MOST IMPORTANT LOGGING SECTION
        self.logger.log_info(f"--- Python backend 'execute' method CALLED. Number of requests in batch: {len(requests)} ---")
        if not requests:
            self.logger.log_error("--- Python backend 'execute' method called with an EMPTY requests list! ---")
            return []

        responses = []
        for i, request_pb in enumerate(requests): # request_pb is of type TritonPythonModel.Request
            self.logger.log_info(f"Processing request_pb {i+1}/{len(requests)}")
            try:

                # Log all available input tensor names from the protobuf request object
                input_tensor_names = [input_tensor.name() for input_tensor in request_pb.inputs()]
                self.logger.log_info(f"Request_pb {i+1}: Available input tensor names: {input_tensor_names}")

                # Attempt to get AUDIO_BYTES and log its presence/absence and basic info
                audio_bytes_tensor = pb_utils.get_input_tensor_by_name(request_pb, "AUDIO_BYTES")
                if audio_bytes_tensor is not None:
                    audio_bytes_content = audio_bytes_tensor.as_numpy()[0] # This should be bytes
                    self.logger.log_info(f"Request_pb {i+1}: Successfully retrieved 'AUDIO_BYTES'. Type: {type(audio_bytes_content)}, Length: {len(audio_bytes_content) if isinstance(audio_bytes_content, bytes) else 'N/A (not bytes)'}")
                else:
                    self.logger.log_error(f"Request_pb {i+1}: FAILED to retrieve 'AUDIO_BYTES' tensor (it's None or not found)!")
                    # Consider returning an error response for this specific request if AUDIO_BYTES is mandatory
                    # responses.append(pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(message="AUDIO_BYTES input is missing")))
                    # continue # Skip to next request in batch

                # Log other critical optional inputs if they are expected to always be there now
                audio_format_tensor = pb_utils.get_input_tensor_by_name(request_pb, "AUDIO_FORMAT")
                if audio_format_tensor is not None:
                    audio_format_str = audio_format_tensor.as_numpy()[0].decode('utf-8')
                    self.logger.log_info(f"Request_pb {i+1}: Successfully retrieved 'AUDIO_FORMAT'. Value: '{audio_format_str}'")
                else:
                    self.logger.log_info(f"Request_pb {i+1}: 'AUDIO_FORMAT' tensor not found by pb_utils. Will use server default: {self.default_audio_format}")

                # ... (Your existing logic to parse other inputs and call _endless_decode_internal) ...
                # Ensure you are passing these parsed values correctly to current_args

                # Example of how you were parsing before, ensure this uses request_pb
                audio_format = audio_format_str if audio_format_tensor is not None else self.default_audio_format
                # ... and so on for other parameters ...

                # --- Your existing execute logic continues here ---
                # This part should now use the variables parsed directly above from request_pb
                class ArgsNamespace(object): pass
                current_args = ArgsNamespace()
                current_args.audio_bytes_data = audio_bytes_content # From above
                current_args.audio_format = audio_format # From above

                # Parse other inputs similarly, falling back to defaults if tensor is None
                total_batch_duration_tensor = pb_utils.get_input_tensor_by_name(request_pb, "TOTAL_BATCH_DURATION_SEC")
                current_args.total_batch_duration = total_batch_duration_tensor.as_numpy()[0] if total_batch_duration_tensor else self.default_total_batch_duration

                chunk_size_tensor = pb_utils.get_input_tensor_by_name(request_pb, "CHUNK_SIZE")
                current_args.chunk_size = chunk_size_tensor.as_numpy()[0] if chunk_size_tensor else self.default_chunk_size

                left_context_size_tensor = pb_utils.get_input_tensor_by_name(request_pb, "LEFT_CONTEXT_SIZE")
                current_args.left_context_size = left_context_size_tensor.as_numpy()[0] if left_context_size_tensor else self.default_left_context_size

                right_context_size_tensor = pb_utils.get_input_tensor_by_name(request_pb, "RIGHT_CONTEXT_SIZE")
                current_args.right_context_size = right_context_size_tensor.as_numpy()[0] if right_context_size_tensor else self.default_right_context_size

                autocast_dtype_str_tensor = pb_utils.get_input_tensor_by_name(request_pb, "AUTOCAST_DTYPE_STR")
                autocast_dtype_str_val = autocast_dtype_str_tensor.as_numpy()[0].decode('utf-8').upper() if autocast_dtype_str_tensor else self.default_autocast_dtype_str.upper()
                current_autocast_dtype = self.autocast_map.get(autocast_dtype_str_val, self.autocast_map["NONE"])

                max_silence_tensor = pb_utils.get_input_tensor_by_name(request_pb, "MAX_SILENCE_FRAMES")
                current_args.max_silence_frames = max_silence_tensor.as_numpy()[0] if max_silence_tensor else self.default_max_silence_frames
                
                self.logger.log_info(f"Request_pb {i+1}: Processing with effective audio_format: {current_args.audio_format}, chunk_size: {current_args.chunk_size}")

                with torch.autocast(self.device.type, dtype=current_autocast_dtype, enabled=(current_autocast_dtype is not None)):
                    transcriptions = self._endless_decode_internal(current_args, self.model, self.char_dict)

                output_tensor = pb_utils.Tensor("TRANSCRIPTIONS", np.array(transcriptions, dtype=object))
                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(inference_response)

            except Exception as e:
                self.logger.log_error(f"Request_pb {i+1}: Exception during execute's main try block: {str(e)}")
                import traceback
                self.logger.log_error(traceback.format_exc())
                error_response = pb_utils.InferenceResponse(output_tensors=[], error=pb_utils.TritonError(message=f"Error processing request: {str(e)}"))
                responses.append(error_response)
                continue

        self.logger.log_info(f"--- Python backend 'execute' method finished. Returning {len(responses)} responses. ---")
        return responses

    def finalize(self):
        """
        Called when the model is unloaded. Perform any cleanup here.
        """
        self.logger.log_info("Model finalized.")