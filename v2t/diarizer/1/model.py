import json
import os
import triton_python_backend_utils as pb_utils
import numpy as np
import nemo.collections.asr as nemo_asr
import torch
import tempfile
import soundfile as sf
import io
from pydub import AudioSegment

class TritonPythonModel:

    # def _load_audio_internal(self, audio_bytes_content):
    #     """
    #     Load audio from raw PCM bytes.
    #     Assumes: 16kHz, mono, 16-bit signed integer PCM (standard for ASR/Diarization).
    #     """
    #     self.logger.log_info(f"Diarizer: Loading audio from raw PCM bytes")
    #     self.logger.log_info(f"Diarizer: Received bytes type: {type(audio_bytes_content)}, length: {len(audio_bytes_content) if hasattr(audio_bytes_content, '__len__') else 'N/A'}")
        
    #     try:
    #         # If audio_bytes_content is bytes, use directly
    #         # If it's a numpy array of bytes, extract the bytes
    #         if isinstance(audio_bytes_content, bytes):
    #             raw_bytes = audio_bytes_content
    #         elif isinstance(audio_bytes_content, np.ndarray):
    #             # Handle case where it's a numpy array containing a bytes object
    #             if audio_bytes_content.dtype == object:
    #                 raw_bytes = audio_bytes_content.item() if audio_bytes_content.size == 1 else audio_bytes_content[0]
    #             else:
    #                 raw_bytes = audio_bytes_content.tobytes()
    #         else:
    #             raw_bytes = bytes(audio_bytes_content)
            
    #         self.logger.log_info(f"Diarizer: Raw bytes length after extraction: {len(raw_bytes)}")
            
    #         # Convert raw bytes directly to numpy array (16-bit signed integers)
    #         audio_data_np = np.frombuffer(raw_bytes, dtype=np.int16)
            
    #         sample_rate = 16000  # Standard rate for ASR/Diarization
            
    #         duration_seconds = len(audio_data_np) / sample_rate
    #         self.logger.log_info(f"Diarizer: Loaded {len(audio_data_np)} samples (~{duration_seconds:.2f}s at {sample_rate}Hz)")
            
    #         if len(audio_data_np) > 0:
    #             self.logger.log_info(f"Diarizer: Sample stats: min={audio_data_np.min()}, max={audio_data_np.max()}, dtype={audio_data_np.dtype}")
    #         else:
    #             raise ValueError("Audio data is empty after conversion")
            
    #         return audio_data_np, sample_rate
            
    #     except Exception as e:
    #         self.logger.log_error(f"Diarizer: Error loading raw PCM audio: {e}")
    #         import traceback
    #         self.logger.log_error(traceback.format_exc())
    #         raise pb_utils.TritonModelException(f"Diarizer: Error loading audio: {e}")
    
    def _load_audio_internal(self, audio_bytes_content, audio_format="wav"): # audio_format is key
        self.logger.log_info(f"Diarizer: Attempting to load audio from in-memory bytes using pydub. Specified format: {audio_format}")
        try:
            audio_file_like_object = io.BytesIO(audio_bytes_content)
            
            audio = AudioSegment.from_file(audio_file_like_object, format=audio_format) # Use the passed format
            self.logger.log_info(f"Diarizer: Loaded from bytes. Duration: {audio.duration_seconds:.2f}s, Channels: {audio.channels}, Frame Rate: {audio.frame_rate}")

            sample_rate = 16000
            audio = audio.set_frame_rate(sample_rate)
            audio = audio.set_sample_width(2)
            audio = audio.set_channels(1)
            self.logger.log_info(f"Diarizer: Resampled/Reformatted. Frame Rate: {audio.frame_rate}, Channels: {audio.channels}")
            
            samples_array = audio.get_array_of_samples()
            samples = torch.as_tensor(samples_array, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            self.logger.log_info(f"Diarizer: Output samples tensor shape: {samples.shape}, dtype: {samples.dtype}, device: {samples.device}")
            if samples.numel() > 0:
                 self.logger.log_info(f"Diarizer: Samples stats: min={samples.min()}, max={samples.max()}")
            return samples_array, sample_rate
        except Exception as e:
            self.logger.log_error(f"Diarizer: Detailed error in _load_audio_internal (pydub from bytes, format: {audio_format}): {e}")
            import traceback; self.logger.log_error(traceback.format_exc())
            raise pb_utils.TritonModelException(f"Diarizer: Error loading audio from in-memory bytes with pydub (format: {audio_format}): {e}")
        

    def initialize(self, args):
        """
        Initialize the model and load the local NeMo diarization model file.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = pb_utils.Logger

        model_repository = args['model_repository']
        model_version = args['model_version']
        model_dir = os.path.join(model_repository, model_version, 'checkpoints')

        nemo_file_name = 'diar_streaming_sortformer_4spk-v2.nemo'
        local_model_path = os.path.join(model_dir, nemo_file_name)

        if not os.path.exists(local_model_path):
            raise pb_utils.TritonModelException(
                f"Nemo model file not found at: {local_model_path}. "
                f"Please make sure '{nemo_file_name}' is in the '{model_dir}' directory."
            )

        self.logger.log_info(f"Loading local NeMo model from: {local_model_path}")

        # Load the Sortformer model
        self.diarizer_model = nemo_asr.models.EncDecDiarLabelModel.restore_from(
            restore_path=local_model_path,
            map_location=self.device
        )
        self.diarizer_model.eval()  # Set to evaluation mode

        self.logger.log_info("Local NeMo Sortformer Diarizer model initialized successfully.")

    def execute(self, requests):
        """
        This function is called for each inference request.
        """
        responses = []
        for request in requests:
            tmp_filename = None
            try:
                audio_bytes_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES")
                if audio_bytes_tensor is not None:
                    audio_bytes_content = audio_bytes_tensor.as_numpy()[0] # This should be bytes
                    self.logger.log_info(f"Successfully retrieved 'AUDIO_BYTES'. Type: {type(audio_bytes_content)}, Length: {len(audio_bytes_content) if isinstance(audio_bytes_content, bytes) else 'N/A (not bytes)'}")
                else:
                    self.logger.log_error(f"FAILED to retrieve 'AUDIO_BYTES' tensor (it's None or not found)!")

                # Load raw PCM audio
                audio_data_np, sample_rate = self._load_audio_internal(audio_bytes_content)

                # if audio_data_np.size == 0:
                #     raise ValueError("Loaded audio data is empty")

                # Write the processed NumPy array to a temporary WAV file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                    sf.write(tmp_audio_file.name, audio_data_np, sample_rate, subtype='PCM_16')
                    tmp_filename = tmp_audio_file.name
                    self.logger.log_info(f"Diarizer: Wrote audio to temp file: {tmp_filename}")

                # Perform diarization
                self.logger.log_info(f"Diarizer: Starting NeMo diarization")
                predicted_segments = self.diarizer_model.diarize(
                    audio=tmp_filename,
                    batch_size=1
                )

                # Convert predicted segments to RTTM format
                self.logger.log_info(f"Diarizer: predicted_segments {predicted_segments}")
                rttm_lines = self._segments_to_rttm(predicted_segments[0])
                self.logger.log_info(f"Diarizer: Generated {len(rttm_lines)} RTTM lines")

                output_tensor = pb_utils.Tensor(
                    "DIARIZATION_RTTM",
                    np.array(rttm_lines, dtype=object)
                )

                inference_response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(inference_response)

            except Exception as e:
                import traceback
                self.logger.log_error(f"Error in diarizer execute: {str(e)}\n{traceback.format_exc()}")
                error = pb_utils.TritonError(message=f"Error in diarizer: {str(e)}")
                responses.append(pb_utils.InferenceResponse(output_tensors=[], error=error))
            finally:
                # Clean up temporary file
                if tmp_filename and os.path.exists(tmp_filename):
                    try:
                        os.unlink(tmp_filename)
                        self.logger.log_info(f"Diarizer: Cleaned up temp file")
                    except Exception as unlink_err:
                        self.logger.log_error(f"Diarizer: Error cleaning up temp file: {unlink_err}")

        return responses

    def _segments_to_rttm(self, predicted_segments_list):
        """
        Convert predicted segments (which are strings) to RTTM format.
        """
        rttm_lines = []
        
        # predicted_segments_list is like: 
        # ['1.200 11.360 speaker_0', '11.600 11.680 speaker_0', ...]
        
        for segment_str in predicted_segments_list:
            try:
                # segment_str is '1.200 11.360 speaker_0'
                parts = segment_str.split()
                if len(parts) != 3:
                    self.logger.log_warn(f"Diarizer: Skipping malformed segment string: {segment_str}")
                    continue

                start_time_str, end_time_str, speaker_label = parts
                
                # Convert to float
                start_time = float(start_time_str)
                end_time = float(end_time_str)
                
                duration = end_time - start_time
                
                # The speaker_label is already 'speaker_0', so use it directly.
                # The old code's int(speaker_id) would have failed.
                rttm_line = f"SPEAKER audio 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_label} <NA> <NA>"
                rttm_lines.append(rttm_line)

            except Exception as e:
                self.logger.log_error(f"Diarizer: Error processing segment string '{segment_str}': {e}")
                pass # Continue to the next segment

        return rttm_lines

    def finalize(self):
        self.logger.log_info("Diarizer model finalized.")