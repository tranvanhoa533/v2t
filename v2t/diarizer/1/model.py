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

    def _load_audio_internal(self, audio_bytes_content, audio_format="wav"): # audio_format is key
        self.logger.log_info(f"Diarizer: Attempting to load audio from in-memory bytes using pydub. Specified format: {audio_format}")
        try:
            audio_file_like_object = io.BytesIO(audio_bytes_content)
            
            audio = AudioSegment.from_file(audio_file_like_object, format=audio_format) # Use the passed format
            self.logger.log_info(f"Diarizer: Loaded from bytes. Duration: {audio.duration_seconds:.2f}s, Channels: {audio.channels}, Frame Rate: {audio.frame_rate}")

            sample_rate = 16000
            audio = audio.set_frame_rate(sample_rate)
            audio = audio.set_sample_width(2) # 16-bit
            audio = audio.set_channels(1) # Mono
            self.logger.log_info(f"Diarizer: Resampled/Reformatted. Frame Rate: {audio.frame_rate}, Channels: {audio.channels}")
            
            samples_array = audio.get_array_of_samples()
            
            # --- ĐÃ SỬA: Chuyển đổi array.array thành numpy.ndarray ---
            samples_np = np.array(samples_array, dtype=np.int16)
            self.logger.log_info(f"Diarizer: Converted pydub array to numpy array. Shape: {samples_np.shape}, dtype: {samples_np.dtype}")
            
            # Trả về numpy array (int16) và sample_rate
            return samples_np, sample_rate
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

        # Đọc model_config để lấy tham số
        self.model_config = json.loads(args['model_config'])

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

        self.diarizer_model = nemo_asr.models.EncDecDiarLabelModel.restore_from(
            restore_path=local_model_path,
            map_location=self.device
        )
        self.diarizer_model.eval()

        # Đọc giá trị mặc định cho AUDIO_FORMAT từ config.pbtxt
        self.default_audio_format = self.model_config['parameters'].get('DEFAULT_AUDIO_FORMAT', {}).get('string_value', 'wav')
        self.logger.log_info(f"Diarizer: Default audio format set to: {self.default_audio_format}")

        self.logger.log_info("Local NeMo Sortformer Diarizer model initialized successfully.")

    def execute(self, requests):
        """
        This function is called for each inference request.
        """
        responses = []
        for request in requests:
            tmp_filename = None
            try:
                # Lấy AUDIO_BYTES (bắt buộc)
                audio_bytes_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_BYTES")
                if audio_bytes_tensor is None:
                    raise pb_utils.TritonModelException("AUDIO_BYTES input is missing")
                
                audio_bytes_content = audio_bytes_tensor.as_numpy()[0]
                self.logger.log_info(f"Diarizer: Received 'AUDIO_BYTES'. Length: {len(audio_bytes_content) if isinstance(audio_bytes_content, bytes) else 'N/A'}")

                # Lấy AUDIO_FORMAT (tùy chọn)
                audio_format_tensor = pb_utils.get_input_tensor_by_name(request, "AUDIO_FORMAT")
                if audio_format_tensor is not None:
                    # Chuyển bytes sang string và lấy chữ thường
                    current_audio_format = audio_format_tensor.as_numpy()[0].decode('utf-8').lower()
                    self.logger.log_info(f"Diarizer: Received 'AUDIO_FORMAT': {current_audio_format}")
                else:
                    current_audio_format = self.default_audio_format
                    self.logger.log_info(f"Diarizer: 'AUDIO_FORMAT' not provided, using default: {current_audio_format}")

                # Tải audio sử dụng định dạng đã xác định
                # 'audio_data_np' BÂY GIỜ sẽ là một numpy array
                audio_data_np, sample_rate = self._load_audio_internal(audio_bytes_content, audio_format=current_audio_format)

                # Dòng này BÂY GIỜ sẽ hoạt động
                if audio_data_np.size == 0:
                    raise ValueError("Loaded audio data is empty")

                # Ghi mảng NumPy (int16) vào file WAV tạm thời
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
                    sf.write(tmp_audio_file.name, audio_data_np, sample_rate, subtype='PCM_16')
                    tmp_filename = tmp_audio_file.name
                    self.logger.log_info(f"Diarizer: Wrote audio to temp file: {tmp_filename}")

                # Thực hiện diarization
                self.logger.log_info(f"Diarizer: Starting NeMo diarization on {tmp_filename}")
                predicted_segments = self.diarizer_model.diarize(
                    audio=tmp_filename,
                    batch_size=1
                )

                # Chuyển đổi segments sang RTTM
                self.logger.log_info(f"Diarizer: predicted_segments {predicted_segments}")
                rttm_lines = self._segments_to_rttm(predicted_segments[0] if predicted_segments else [])
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
                # Dọn dẹp file tạm
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
        
        for segment_str in predicted_segments_list:
            try:
                parts = segment_str.split()
                if len(parts) != 3:
                    self.logger.log_warn(f"Diarizer: Skipping malformed segment string: {segment_str}")
                    continue

                start_time_str, end_time_str, speaker_label = parts
                
                start_time = float(start_time_str)
                end_time = float(end_time_str)
                duration = end_time - start_time
                
                if duration < 0:
                     self.logger.log_warn(f"Diarizer: Skipping segment with negative duration: {segment_str}")
                     continue

                rttm_line = f"SPEAKER audio 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_label} <NA> <NA>"
                rttm_lines.append(rttm_line)

            except Exception as e:
                self.logger.log_error(f"Diarizer: Error processing segment string '{segment_str}': {e}")
                pass 

        return rttm_lines

    def finalize(self):
        self.logger.log_info("Diarizer model finalized.")